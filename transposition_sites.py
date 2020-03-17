#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
@author: Markus Haak (markus.haak@posteo.net)
https://github.com/MarkusHaak/Radukic_Brandt_2020

This file is part of the repository Radukic_Brandt_2020. The scripts in Radukic_Brandt_2020 are free software: 
You can redistribute it and/or modify them under the terms of the GNU General Public License as published by the 
Free Software Foundation, either version 3 of the License, or (at your option) any later version. Radukic_Brandt_2020
scripts are distributed in the hope that they will be useful, but WITHOUT ANY WARRANTY; without even the implied 
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
details. You should have received a copy of the GNU General Public License along with mega. If not, 
see <http://www.gnu.org/licenses/>.
"""

import os, sys, argparse
from progress.bar import Bar
from Bio import SeqIO
from Bio.Seq import Seq
import numpy as np
from multiprocessing import Pool
from collections import namedtuple
import re
import random

class ArgHelpFormatter(argparse.HelpFormatter):
    '''
    Formatter adding default values to help texts.
    '''
    def __init__(self, prog):
        super().__init__(prog)

    def _get_help_string(self, action):
        text = action.help
        if  action.default is not None and \
            action.default != argparse.SUPPRESS and \
            'default:' not in text.lower():
            text += ' (default: {})'.format(action.default)
        return text

token_specification = [
        ('DELETION',  r'(\d+)D'), 
        ('HCLIPPING', r'(\d+)H'), 
        ('INSERTION', r'(\d+)I'), 
        ('MATCH',     r'(\d+)M'), 
        ('SKIPPED',   r'(\d+)N'), 
        ('PADDING',   r'(\d+)P'), 
        ('SCLIPPING', r'(\d+)S'), 
        ('MISMATCH',  r'(\d+)X'), 
        ('READMATCH', r'(\d+)=')  
    ]
tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)

PAFcontent = namedtuple('PAFcontent', 'qid qlen q_st q_en strand ctg ctg_len r_st r_en mlen blen mapq cigar_str others')

def parse_args():
    parser = argparse.ArgumentParser(description='Estimates transposition sites for ONT rapid library sequencing runs.',
                                     formatter_class=ArgHelpFormatter, 
                                     add_help=False)

    main_group = parser.add_argument_group('Main Options')
    main_group.add_argument('reads',
                            nargs='+',
                            help='Fastq files or path to directories containing fastq files (recursion depth 1).')
    main_group.add_argument('chromosome',
                            help='Fasta file containing the genomic sequences that is searched for insertion sites.')
    main_group.add_argument('adapter',
                            help='Transposon Y adapter sequence.')
    main_group.add_argument('--paf',
                            help='Initial alignment file.')
    main_group.add_argument('--out',
                            help="Basename for saving transposition sites in tap seperated value (.tsv) format. "+\
                                 "Sites on the plus and minus strand of the chromosome are saved in seperated files, "+\
                                 "replacing a set of braces in the given Basename with plus or minus, respectively.",
                            default="transposition_sites_{}.tsv")
    main_group.add_argument('--circular',
                            action="store_true")
    main_group.add_argument('--min_blen',
                            help="Minimal length of a produced alignment (including errors) to be considered in "+\
                                 "the analysis.",
                            type=int,
                            default=100)
    main_group.add_argument('--chr_seq_len',
                            help="Length of the chromosome sequence downstream of the adapter sequence. "+\
                                 "Should be <= (min_blen - max_diff)",
                            type=int,
                            default=75)
    main_group.add_argument('--max_diff',
                            help="Max diversion of the refined to the initial transposition site estimate. "+\
                                 "Typically, the mean difference is approx. 3-5 nt for rapid kits (std. div. 3-5 nt).",
                            type=int,
                            default=10)
    main_group.add_argument('--min_readlength',
                            help="Minimal length of reads to be considered in the analysis.",
                            type=int,
                            default=500)
    main_group.add_argument('--max_q_st',
                            help="Max distance of the alignment start to the read start. Useful as a filter prior to "+\
                                 "transposition site refinement.",
                            type=int,
                            default=None)
    main_group.add_argument('--alignment_criterium',
                            help="Defines which criterium is being used for selecting from multiple alignments of a "+\
                                 "single read in the initial alignment step",
                            choices=["best", "longest", "close_to_q_st"],
                            default="close_to_q_st")
    main_group.add_argument('--minimap2',
                            help="Specify a differnet minimap2 executable than the one in the PATH variable.",
                            default='minimap2')
    main_group.add_argument('--processes',
                            help="Number of processes used for multiprocessing.",
                            type=int,
                            default=6)

    filter_group = parser.add_argument_group('Filter Options')
    filter_group.add_argument('--filter_window',
                              help="Sequence window around the position of transition from "+\
                                   "the adapter sequence to the chromosome sequence.",
                              type=int,
                              default=10)
    filter_group.add_argument('--filter_max_err',
                              help="Maximum amount of errors (insertions, deletions) "+\
                                   "within the specified sequence window.",
                              type=int,
                              default=3)
    filter_group.add_argument('--filter_print',
                              help="Fraction of randomly selected realignments to be printed to stdout.",
                              type=float,
                              default=0)
    
    help_group = parser.add_argument_group('Help')
    help_group.add_argument('-h', '--help', 
                            action='help', 
                            default=argparse.SUPPRESS,
                            help='Show this help message and exit.')

    return parser.parse_args()


def main():
    print("reading adapter sequence")
    for record in SeqIO.parse(args.adapter, "fasta"):
        adapter_seq = record.seq

    print("reading chromosome fasta")
    chromosome = {}
    for record in SeqIO.parse(args.chromosome, "fasta"):
        chromosome[record.id] = record

    fq_files = []
    for entry in args.reads:
        if os.path.isfile(entry) and entry.endswith(".fastq"):
            fq_files.append(entry)
        else:
            fq_files.extend([os.path.join(entry, f) for f in os.listdir(entry) \
                             if os.path.isfile(os.path.join(entry, f)) and f.endswith(".fastq")])

    print("reading fastq records")
    reads = {}
    bar = Bar('progress', max=len(fq_files))
    for fqFile in fq_files:
        for record in SeqIO.parse(fqFile, "fastq"):
            reads[record.id] = record
        bar.next()
    bar.finish()

    if args.paf:
        paf_fn = args.paf
    else:
        print("creating initial alignment for transposition site estimates")
        fq_fn = "tmp_fw_reads.fasta"
        ref_fn = args.chromosome
        paf_fn = "tmp_alignment.paf"
        os.system('{} -x map-ont --eqx -t 4 {} {} >{} 2> ./minimap2_messages'\
                  .format(args.minimap2, ref_fn, fq_fn, paf_fn))
    alignments = parse_paf(paf_fn, keep=args.alignment_criterium)

    transp_sites_plus = []
    transp_sites_minus = []
    transp_sites_plus_rid = []
    transp_sites_minus_rid = []
    q_st_plus = []
    q_st_minus = []
    for qid,ctg in alignments:
        hit = alignments[(qid,ctg)]
        if hit.qlen >= args.min_readlength:
            if args.max_q_st is not None:
                if hit.q_st > args.max_q_st:
                    continue
            if hit.strand == 1:
                transp_sites_plus.append(hit.r_st)
                transp_sites_plus_rid.append(hit.qid)
                q_st_plus.append(hit.q_st)
            else:
                transp_sites_minus.append(hit.r_en - 1) 
                transp_sites_minus_rid.append(hit.qid)
                q_st_minus.append(hit.q_st)

    transp_sites_plus_bins = {}
    transp_sites_minus_bins = {}
    for rs,qid in zip(transp_sites_plus,transp_sites_plus_rid):
        if rs in transp_sites_plus_bins:
            transp_sites_plus_bins[rs].append(qid)
        else:
            transp_sites_plus_bins[rs] = [qid]
    for rs,qid in zip(transp_sites_minus,transp_sites_minus_rid):
        if rs in transp_sites_minus_bins:
            transp_sites_minus_bins[rs].append(qid)
        else:
            transp_sites_minus_bins[rs] = [qid]

    chrm = chromosome[list(chromosome.keys())[0]]
    transp_sites_plus, transp_sites_minus = refine_transp_sites(adapter_seq, alignments, chrm, reads, 
                                                             transp_sites_plus_bins, transp_sites_minus_bins)

    chrom = list(chromosome.keys())[0]
    if args.out:
        if "{}" not in args.out:
            l = args.out.split(".")
            l.insert(-2, "_{}")
            args.out = ".".join(l)
        with open(args.out.format("plus"), "w") as f:
            for i in range(len(chromosome[chrom].seq)):
                print("{}\t{}".format(i+1, transp_sites_plus.count(i)), file=f)
        with open(args.out.format("minus"), "w") as f:
            for i in range(len(chromosome[chrom].seq)):
                print("{}\t{}".format(i+1, transp_sites_minus.count(i)), file=f)
    else:
        print("\nPlus strand:")
        for i in range(len(chromosome[chrom].seq)):
            print("{}\t{}".format(i+1, transp_sites_plus.count(i)))
        print("\nMinus strand:")
        for i in range(len(chromosome[chrom].seq)):
            print("{}\t{}".format(i+1, transp_sites_minus.count(i)))

def expand_cigars(hit, qseq, rseq):
    s = hit.r_st
    q = hit.q_st
    s_str, m_str, q_str = "", "", ""

    for mo in re.finditer(tok_regex, hit.cigar_str):
        kind = mo.lastgroup
        value = int(mo.group(0)[:-1])
        if kind == 'DELETION':
            s_str += rseq[s:s+value]
            m_str += " "*value
            q_str += "-"*value
            s += value
        elif kind == 'INSERTION':
            s_str += "-"*value
            m_str += " "*value
            q_str += qseq[q:q+value]
            q += value
        elif kind == 'MATCH' or kind == 'READMATCH':
            s_str += rseq[s:s+value]
            m_str += "|"*value
            q_str += qseq[q:q+value]
            s += value
            q += value
        elif kind == 'MISMATCH':
            for i in range(value):
                s_str += rseq[s]
                m_str += "X"
                q_str += qseq[q]
                q += 1
                s += 1
        else:
            print("NOT HANDLED")
    return s_str, m_str, q_str

def refine_transp_sites(adapter_seq, initial_alignments, chrm, reads, transp_sites_plus, transp_sites_minus):
    """
        Refines estimated transposition sites by realigning reads against a set of reference sequences consisting of
        the sequencing adapter and a genomic sequence starting at positions neighboring the initial estimation.
    """

    def quality_check(hit, qseq, rseq):
        """
            Parses an alignment's cigar string to check if there are no more than the accepted amount of insertions or
            deletions in a specified window around the estimated transposition site.

            Returns
            -------
            Boolean
                True if the alignment passes the quality check, False otherwise
        """
        window = set(range(len(adapter_seq)-args.filter_window//2,len(adapter_seq)+args.filter_window//2))
        s = hit.r_st
        q = hit.q_st
        errors = 0
        total_gaps = 0
        for mo in re.finditer(tok_regex, hit.cigar_str):
            kind = mo.lastgroup
            value = int(mo.group(0)[:-1])
            for i in range(value):
                if kind == 'DELETION':
                    if s in window:
                        errors += 1
                    total_gaps += 1
                    s += 1
                elif kind == 'INSERTION':
                    if s in window:
                        errors += 1
                    total_gaps += 1
                    q += 1
                elif kind == 'MATCH' or kind == 'READMATCH':
                    s += 1
                    q += 1
                elif kind == 'MISMATCH':
                    q += 1
                    s += 1
                else:
                    print("NOT HANDLED")
        if errors > args.filter_max_err:
            return False
        return True

    def dist(a,b):
        """
            Returns the signed distance from chromosome site a to site b, given that
            the chromosome is either circular or linear.

            Parameters
            ----------
            a: int
                The minuend site.
            b: int
                The subtrahend site.
                
            Returns
            -------
            int
                A signed integer that is the equivalent of a - b.
        """
        if a > b:
            if args.circular:
                if (a-b) < ((len(chrm.seq)+b) - a):
                    return a - b
                else:
                    return a - (len(chrm.seq)+b)
            else:
                return a - b
        elif a < b:
            if args.circular:
                if (b-a) < ((len(chrm.seq)+a) - b):
                    return a - b
                else:
                    return (len(chrm.seq)+a) - b
            else:
                return a - b
        else:
            return 0

    def dist_hit(hit, init_hit):
        """
            Returns the distance in nucleotides between the initial and the refined transposition site estimate. 
        """
        init_transp_site = init_hit.r_st if init_hit.strand == 1 else init_hit.r_en -1
        refined_transp_site = int(hit.ctg)
        return dist(refined_transp_site, init_transp_site)

    def print_comparison(hit, init_hit, rseqs):
        """
            Prints a read sequence aligned to the reference sequence (sequencing adapter + chromosomal sequence) for
            the refined transposition site and the initial estimate for comparison.
        """
        print()
        init_transp_site = init_hit.r_st if init_hit.strand == 1 else init_hit.r_en -1
        ori = "plus" if init_hit.strand == 1 else "minus"
        refined_transp_site = int(hit.ctg)
        print(ori, ":", init_transp_site, "-->", refined_transp_site, "({})".format(hit.qid))
        qseq = reads[hit.qid].seq
        rseq = rseqs[refined_transp_site]
        s_str, m_str, q_str = expand_cigars(hit, qseq, rseq)
        passed = quality_check(hit, qseq, rseq)
        print("\n NEW:", hit.mapq, passed)
        blocks = round((len(m_str) / 78.) + 0.5)
        for i in range(blocks):
            print(s_str[i*78:(i+1)*78])
            print(m_str[i*78:(i+1)*78])
            print(q_str[i*78:(i+1)*78])
            print()
        prev_hit = None
        for hit_ in hit.others:
            if int(hit_.ctg) == init_transp_site:
                prev_hit = hit_
        if prev_hit:
            qseq = reads[prev_hit.qid].seq
            rseq = rseqs[int(prev_hit.ctg)]
            s_str, m_str, q_str = expand_cigars(prev_hit, qseq, rseq)
            passed = quality_check(prev_hit, qseq, rseq)
            print("\n PREVIOUS:", prev_hit.mapq, passed)
            blocks = round((len(m_str) / 78.) + 0.5)
            for i in range(blocks):
                print(s_str[i*78:(i+1)*78])
                print(m_str[i*78:(i+1)*78])
                print(q_str[i*78:(i+1)*78])
                print()

    refined_transp_sites_plus = []
    refined_transp_sites_minus = []
    for orientation, transp_sites in [(1,transp_sites_plus), (-1, transp_sites_minus)]:
        ori_str = "plus" if orientation == 1 else "minus"
        ori_reads = 0
        for site in transp_sites:
            ori_reads += len(transp_sites[site])
        realigned_read_cnt = 0
        distances = []
        refined_transp_sites = []

        print("refining {} {} reads".format(ori_reads, ori_str))
        bar = Bar('progress', max=len(transp_sites))
        for site in transp_sites:
            multi_ref_fn,rseqs = make_multi_reference_fasta(adapter_seq, chrm, [site], orientation)
            paf_fn = "tmp_realignment.paf"
            fq_fn = "tmp_queries.fastq"
            records = [reads[qid] for qid in transp_sites[site]]
            with open(fq_fn, "w") as f:
                SeqIO.write(records, f, "fasta")
            os.system('{} --eqx -x map-ont --for-only -c -t 4 {} {} >{} 2> ./minimap2_messages'\
                      .format(args.minimap2, multi_ref_fn, fq_fn, paf_fn))
            alignments = parse_paf_realignment(paf_fn)
            realigned_read_cnt += len(alignments)

            for qid in alignments:
                hit = alignments[qid]
                init_hit = initial_alignments[(qid,chrm.id)]
                qseq = reads[hit.qid].seq
                rseq = rseqs[int(hit.ctg)]
                passed = quality_check(hit, qseq, rseq)
                if passed:
                    distances.append(dist_hit(hit, init_hit))
                    refined_transp_sites.append(int(hit.ctg))
                if args.filter_print:
                    if random.random() < args.filter_print:
                        print_comparison(hit, init_hit, rseqs)
            bar.next()
        bar.finish()
        print("realigned:                  ", realigned_read_cnt)
        print("passed:                     ", len(refined_transp_sites))
        print("mean trasp. site correction:", np.mean(distances))
        print("standard deviation:         ", np.std(distances))
        if orientation == 1:
            refined_transp_sites_plus = refined_transp_sites[:]
        else:
            refined_transp_sites_minus = refined_transp_sites[:]

    return refined_transp_sites_plus, refined_transp_sites_minus

def make_multi_reference_fasta(adapter_seq, chrm, transp_sites, orientation):
    """
        Creates the set of reference sequences for transposition site refinement, saves them in a multiple fasta file
        and returns the filename as well as a list of the sequences themselves.
    """
    adapter_seq = adapter_seq.upper()
    fn = "tmp_multi_ref_{}.fasta".format(orientation)
    sites = set()
    for i in transp_sites:
        for site in range(i-args.max_diff, i+args.max_diff+1):
            if orientation == 1:
                if (site + args.chr_seq_len) >= len(chrm.seq):
                    if args.circular:
                        if site >= len(chrm.seq):
                            sites.add(site - len(chrm.seq))
                        else:
                            sites.add(site)
                elif site < 0:
                    if args.circular:
                        sites.add(site + len(chrm.seq))
                else:
                    sites.add(site)
            elif orientation == -1:
                if site >= len(chrm.seq):
                    if args.circular:
                        sites.add(site - len(chrm.seq))
                elif (site - args.chr_seq_len+1) < 0:
                    if args.circular:
                        if site < 0:
                            sites.add(site + len(chrm.seq))
                        else:
                            sites.add(site)
                else:
                    sites.add(site)
            else:
                print("orientation must be either 1 or -1")
                exit(1)
    rseqs = {}
    with open(fn, "w") as f:
        if orientation == 1:
            for site in sites:
                if (site + args.chr_seq_len) >= len(chrm.seq):
                    seq = adapter_seq + str((chrm.seq+chrm.seq[:args.chr_seq_len])[site:site+args.chr_seq_len]).lower()
                else:
                    seq = adapter_seq + str(chrm.seq[site:site+args.chr_seq_len]).lower()
                rseqs[site] = seq
                print(">{}\n{}".format(site, seq), file=f)
        else:
            for site in sites:
                if (site - args.chr_seq_len) < 0:
                    seq = adapter_seq + str((chrm.seq[-args.chr_seq_len:]+chrm.seq)[site+1:site+args.chr_seq_len+1].reverse_complement()).lower()
                else:
                    seq = adapter_seq + str(chrm.seq[site-args.chr_seq_len+1:site+1].reverse_complement()).lower()
                rseqs[site] = seq
                print(">{}\n{}".format(site,seq), file=f)
    return fn, rseqs

def parse_paf_realignment(fn):
    def int_or_str(s):
        if s.isdigit():
            return int(s)
        elif s in "+-":
            return int(s+"1")
        return s

    alignments = {}
    with open(fn, "r") as f:
        for line in f.readlines():
            line = line.strip()
            try:
                cigar_str = re.search('\scg:Z:(\S*)', line).group(1)
            except:
                cigar_str = None
            hit = PAFcontent( *[int_or_str(s) for s in line.split()[:12]], cigar_str, [] )
            if hit.blen >= args.min_blen:
                if hit.qid in alignments:
                    if hit.mapq > alignments[hit.qid].mapq:
                        hit.others.append(alignments[hit.qid])
                        alignments[hit.qid] = hit
                    else:
                        alignments[hit.qid].others.append(hit)
                else:
                    alignments[hit.qid] = hit
    return alignments

def parse_paf(fn, keep="close_to_q_st"):
    def int_or_str(s):
        if s.isdigit():
            return int(s)
        elif s in "+-":
            return int(s+"1")
        return s

    alignments = {}
    with open(fn, "r") as f:
        for line in f.readlines():
            line = line.strip()
            try:
                cigar_str = re.search('\scg:Z:(\S*)', line).group(1)
            except:
                cigar_str = None
            hit = PAFcontent( *[int_or_str(s) for s in line.split()[:12]], cigar_str, [] )
            if hit.blen >= args.min_blen:
                if (hit.qid,hit.ctg) in alignments:
                    if keep == "best":
                        if hit.mapq > alignments[(hit.qid,hit.ctg)].mapq:
                            alignments[(hit.qid,hit.ctg)] = hit
                    if keep == "longest":
                        if alignments[(hit.qid,hit.ctg)].len < hit.len:
                            alignments[(hit.qid,hit.ctg)] = hit
                    elif keep == "close_to_q_en":
                        if alignments[(hit.qid,hit.ctg)].q_en < hit.q_en:
                            alignments[(hit.qid,hit.ctg)] = hit
                    elif keep == "close_to_q_st":
                        if hit.q_st < alignments[(hit.qid,hit.ctg)].q_st:
                            alignments[(hit.qid,hit.ctg)] = hit
                else:
                    alignments[(hit.qid,hit.ctg)] = hit
    return alignments

if __name__ == '__main__':
    args = parse_args()
    main()
