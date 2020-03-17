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
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from matplotlib import mlab as ml
from matplotlib import colors as colors
from matplotlib.colors import LogNorm
from matplotlib.widgets import LassoSelector
from matplotlib.widgets import EllipseSelector
from matplotlib.path import Path
from matplotlib.collections import PathCollection
from matplotlib.markers import MarkerStyle
import matplotlib.gridspec as gridspec
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from Bio import SeqIO
from Bio.Seq import Seq
import numpy as np
from collections import namedtuple
import random
from progress.bar import Bar
import copy

PAFcontent = namedtuple('PAFcontent', 'qid qlen q_st q_en strand ctg ctg_len r_st r_en mlen blen mapq cigar_str others joined circularized')
Read = namedtuple('Read', "id len tlen seq_5 seq_3_rc pids_5 pos_5 pids_3 pos_3")

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

#Adapted version of https://matplotlib.org/3.1.0/gallery/widgets/lasso_selector_demo_sgskip.html
class SelectFromCollection(object):
    """
        Select data points from a matplotlib collection using `LassoSelector`.
    """

    def __init__(self, ax, collections, alpha_other=0.1):
        self.canvas = ax.figure.canvas
        self.collections = collections
        self.alpha_other = alpha_other
        self.shift = False
        self.path = None

        self.xys = [collection.get_offsets() for collection in collections]

        self.fc = []
        for i,collection in enumerate(collections):
            self.fc.append(collection.get_facecolors())
            if len(self.fc[i]) == 0:
                raise ValueError('Collection must have a facecolor')
            elif len(self.fc[i]) == 1 and len(self.xys[i]) > 1:
                self.fc[i] = np.tile(self.fc[i], (len(self.xys[i]), 1))

        self.selector = LassoSelector(ax, onselect=self.onselect_lasso)
        self.ind = [[] for i in range(len(self.collections))]

    def onselect_lasso(self, verts):
        self.path = Path(verts, closed=True)
        for i in range(len(self.collections)):
            self.ind[i] = np.nonzero(self.path.contains_points(self.xys[i]))[0]
            self.fc[i][:, -1] = self.alpha_other
            self.fc[i][self.ind[i], -1] = min(1.0, 2*self.alpha_other)
            self.collections[i].set_facecolors(self.fc[i])
        self.canvas.draw_idle()

    def onselect_ellipse(self, eclick, erelease):
        xcenter, ycenter, width, height = self.selector._rect_bbox
        e = mpatches.Ellipse([xcenter+(width/2.), ycenter+(height/2.)], width, height, 
                             linewidth=1, fill=False, zorder=2)
        self.path = e.get_transform().transform_path(e.get_path())
        for i in range(len(self.collections)):
            self.ind[i] = np.nonzero(self.path.contains_points(self.xys[i]))[0]
            self.fc[i][:, -1] = self.alpha_other
            self.fc[i][self.ind[i], -1] = min(1.0, 2*self.alpha_other)
            self.collections[i].set_facecolors(self.fc[i])
        self.canvas.draw_idle()

    def accept(self):
        for i in range(len(self.collections)):
            self.fc[i][self.ind[i], -1] = self.alpha_other
            self.collections[i].set_facecolors(self.fc[i])
        self.canvas.draw_idle()
        return self.path

    def disconnect(self):
        self.selector.disconnect_events()
        for i in range(len(self.collections)):
            self.fc[i][:, -1] = self.alpha_other
            self.collections[i].set_facecolors(self.fc[i])
        self.canvas.draw_idle()

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize sstart, send, qstart and qend of plus and minus alignments \
                                                  in a paf file as 2D dotplots.',
                                     formatter_class=ArgHelpFormatter, 
                                     add_help=False)

    main_group = parser.add_argument_group('General Options')
    main_group.add_argument('paf_fn',
                            help='Path to a .paf file created with minimap2 and option secondary=no.')
    main_group.add_argument('-o', '--out',
                            help='Filename scheme under which to save the produced plot(s). A single pair of {} is \
                                  replaced by the respective subject id. File type is respected (png, svg, pdf, ...). \
                                  (default: Show on screen only)')
    main_group.add_argument('-x', '--xlim',
                            help='Lower and upper limits of the x axis. Both must be given.',
                            nargs=2,
                            default=None)
    main_group.add_argument('-y', '--ylim',
                            help='Lower and upper limits of the y axis. Both must be given.',
                            nargs=2,
                            default=None)

    data_group = parser.add_argument_group('Data options')
    data_group.add_argument('--subjects',
                            help='The subject(s) for which an individual plot will be produced. (default: all)',
                            nargs='+',
                            default=[])
    data_group.add_argument('--circular',
                            help='If set, two non-overlapping primary alignments of a single read will be joint if they\
                                  span an edge of the subject sequence and are no more than {--max_dist} apart from \
                                  each other, both in terms of subject sequence and query sequence. This is done prior \
                                  to other data filters.',
                            action='store_true')
    data_group.add_argument('--max_dist',
                            help='Maximal distance that two alignments of a single read are allowed to be apart for \
                                  them to be joint to a single alignment, given that --circular is specified.',
                            type=int,
                            default=100)
    data_group.add_argument('--min_qlen',
                            help='Minimal query sequence length (read length).',
                            type=int)
    data_group.add_argument('--max_qst',
                            help="Maximal length of the non-aligned 5' overhang. All alignments with a longer 5' \
                                  do not contribute to the produced plots.",
                            type=int)
    data_group.add_argument('--max_qen',
                            help="Maximal length of the non-aligned 3' overhang. All alignments with a longer 53' \
                                  do not contribute to the produced plots.",
                            type=int)
    data_group.add_argument('--multiple',
                            help='Sets which entries to plot in case of several (primary, non-overlapping) alignments \
                                  of a single read against one subject sequence. Does not affect multiple alignments \
                                  against different subjects.',
                            choices=["all", "none", "joined", "best", "longest", "close_to_q_st", "close_to_q_en"],
                            default='none')
    data_group.add_argument('--project',
                            help="Instead of plotting the lengths of 5' and 3' non-aligning sequences in seperate \
                                  subplots above and right of the central 2D dotplot, they are added to the \
                                  produced alignments as if they would perfectly align to the subject sequence. \
                                  Please combine with option 'no_overhangs'.",
                            action="store_true")
    data_group.add_argument('--no_overhangs',
                            help="Only plot the central 2D dotplot and not the subplots above and right of it that \
                                  display non-aligned overhang lengths.",
                            action='store_true')
    data_group.add_argument('--subfigures',
                            help='Defines which of the two subfigures for plus and minus alignments are being plotted.',
                            choices=['both', 'plus', 'minus'],
                            default='both')

    over_group = parser.add_argument_group('Sampling and Overplotting')
    over_group.add_argument('--subsample',
                            help="Of all alignments against a given subject, only plot a randomly chosen subsample of \
                            the given size. This is applied after all data handling steps.",
                            type=int)
    over_group.add_argument('-e','--equal',
                            help="Select an equal number of reads mapping to the plus and minus strand \
                                  ({--sumsample} // 2).",
                            action="store_true")
    over_group.add_argument('--seed',
                            help='Seed for the python library random to reproduce a previous results.',
                            type=int)
    over_group.add_argument('-a', '--alpha',
                            help="The alpha value of the plotted data points.",
                            default=0.1,
                            type=float)
    over_group.add_argument('-s', '--surface',
                            help="The surface area or plotted data points.",
                            default=1.0,
                            type=float)
    over_group.add_argument('--hist2d',
                            help="Plot as a 2d Histogram with the specified bin width.",
                            type=int)
    over_group.add_argument('--log',
                            help="Plot the 2d Histogram with log scale colors.",
                            action="store_true")

    high_group = parser.add_argument_group('Highlighting Options')
    high_group.add_argument('-p','--primers',
                            help="Fasta file containing primer sequences that shall be detected at the start of the \
                                  reads and their reverse complement at the reads' ends.")
    high_group.add_argument('--annotation',
                            help="Tab saperated values file (without header) containing read ids in the first column, \
                                  space seperated 5' primer ids and their start positions in the second column and \
                                  third column respectively.")
    high_group.add_argument('--hide',
                            help='Whether to hide data points of reads that could not be classified. \
                                  (default: plot everything)',
                            choices=['all_unclassified', "5'_unclassified", "3'_unclassified", "all_classified", 
                                     "5'_classified", "3'_classified"])
    high_group.add_argument('--primer_as_terminal',
                            help='Instead of qst and/or (qlen-qen), display the distance of detected primers from the \
                                  mapping part of each read.',
                            action="store_true")
    high_group.add_argument('--subfigure',
                            help='Character for identification of the created plot as a subfigure of a greater plot.')
    high_group.add_argument('--interactive',
                            action='store_true')

    styling_group = parser.add_argument_group('Styling Options')
    styling_group.add_argument('--dpi',
                              help='The resolution of the saved image file.',
                              type=int,
                              default=72)
    styling_group.add_argument('--width',
                              help='The width in inches of the produced image.',
                              type=float,
                              default=20)
    styling_group.add_argument('--height',
                              help='The height in inches of the produced image.',
                              type=float,
                              default=8)
    styling_group.add_argument('--fontsize',
                              help='The fontsize for all texts.',
                              type=int,
                              default=9)
    styling_group.add_argument('--spacing',
                              help='Spacing to the left, right, top, bottom and in between the two subfigures.',
                              type=float,
                              nargs=5,
                              default=[0.01, 0.95, 0.95, 0.08, 0.3])
    styling_group.add_argument('--subplot_sizes',
                              help='Sizes of the top and right subplots of each subfigure as well as the colormap \
                                    in relation to the central subplot size.',
                              type=float,
                              nargs=3,
                              default=[0.2, 0.3, 0.05])
    styling_group.add_argument('--spines_linewidth',
                              help='linewidth of all spines in the produced image.',
                              type=float,
                              default=1.0)
    styling_group.add_argument('--colormap',
                              help='',
                              choices=["YlOrRd", "Purples"],
                              default="YlOrRd")
    
    help_group = parser.add_argument_group('Help')
    help_group.add_argument('-h', '--help', 
                            action='help', 
                            default=argparse.SUPPRESS,
                            help='Show this help message and exit.')

    args = parser.parse_args()

    if args.primer_as_terminal and not args.hide in ['all_unclassified', "5'_unclassified", "3'_unclassified"]:
        args.hide = "all_unclassified"

    if args.annotation:
        print("parsing annotation file")
        reads = parse_annotation_file(args.annotation)

    if not args.seed:
        args.seed = random.randrange(sys.maxsize)

    if args.colormap == "YlOrRd":
        args.colormap = cm.YlOrRd
    elif args.colormap == "Purples":
        args.colormap = cm.Purples

    return args

def main():
    rng = random.Random(args.seed)
    print("random seed set to {}".format(args.seed))

    print("parsing paf file {}".format(args.paf_fn))
    paf_content = parse_paf(args.paf_fn)
    all_alignments = preprocess_data(paf_content, args.subjects, args.circular, args.max_dist, args.multiple, 
                                     args.min_qlen, args.max_qst, args.max_qen, args.project)

    for subject in all_alignments:
        print("Processing subject {}".format(subject))
        alignments = all_alignments[subject]

        num_alignments = {1 :len([i for i in alignments if i.strand == 1]),
                          -1:len([i for i in alignments if i.strand == -1])}
        num_joined = {1 :len([i for i in alignments if i.strand == 1 and i.joined]),
                      -1:len([i for i in alignments if i.strand == -1 and i.joined])}
        num_circularized = {1 :len([i for i in alignments if i.strand == 1 and i.circularized]),
                            -1:len([i for i in alignments if i.strand == -1 and i.circularized])}
        print("\n{:>12} plus alignments in total".format(num_alignments[1]))
        print("{:>12} circularized plus alignments".format(num_circularized[1]))
        print("{:>12} joined plus alignments".format(num_joined[1]))
        print("\n{:>12} minus alignments in total".format(num_alignments[-1]))
        print("{:>12} circularized minus alignments".format(num_circularized[-1]))
        print("{:>12} joined minus alignments".format(num_joined[-1]))
        print()

        if args.subsample:
            if args.equal:
                alignments_plus = [hit for hit in alignments if hit.strand == 1]
                alignments_minus = [hit for hit in alignments if hit.strand == -1]
                samplesize = min([args.subsample // 2, len(alignments_plus), len(alignments_minus)])
                alignments = random.sample(alignments_plus, samplesize) + random.sample(alignments_minus, samplesize)
            else:
                samplesize = min([args.subsample, len(alignments)])
                alignments = random.sample(alignments, samplesize)

        num_selection = {1 :len([i for i in alignments if i.strand == 1]),
                         -1:len([i for i in alignments if i.strand == -1])}
        num_joined = {1 :len([i for i in alignments if i.strand == 1 and i.joined]),
                      -1:len([i for i in alignments if i.strand == -1 and i.joined])}
        num_circularized = {1 :len([i for i in alignments if i.strand == 1 and i.circularized]),
                            -1:len([i for i in alignments if i.strand == -1 and i.circularized])}
        print("\n{:>12} plus alignments in selection".format(num_selection[1]))
        print("{:>12} circularized plus alignments".format(num_circularized[1]))
        print("{:>12} joined plus alignments".format(num_joined[1]))
        print("\n{:>12} minus alignments in selection".format(num_selection[-1]))
        print("{:>12} circularized minus alignments".format(num_circularized[-1]))
        print("{:>12} joined minus alignments".format(num_joined[-1]))
        print()
        

        if args.annotation:
            primer_ids = []
            primer_seqs = []
            primer_colors = {}
            unclassified_color = "C0"
            multiple_color = "C0"
            i = 2
            for record in SeqIO.parse(args.primers, "fasta"):
                primer_ids.append(record.id)
                primer_seqs.append(str(record.seq).upper())
                primer_colors[record.id] = "C{}".format(i)
                i += 1

        if not args.xlim:
            args.xlim = [min([hit.r_st for hit in alignments]), max([hit.r_en for hit in alignments])]
        else:
            args.xlim = [int(args.xlim[0]), int(args.xlim[1])]
        if not args.ylim:
            args.ylim = args.xlim
        else:
            args.ylim = [int(args.ylim[0]), int(args.ylim[1])]

        print("\nplotting ...", end="")

        plt.rcParams.update({'font.size': args.fontsize, 'axes.linewidth': args.spines_linewidth})
        fontname = {'fontname':'Arial'}

        fig = plt.figure(figsize=(args.width, args.height), dpi=args.dpi)

        left, right, top, bottom, center = args.spacing
        topplots_hight, rightplots_width, cb_width = args.subplot_sizes
        cb_dist = 0.02

        if args.subfigure:
            subfigure_str_width = 0.2
        else:
            subfigure_str_width = 0.


        left_main_width = 1.
        right_main_width = 1.
        if args.subfigures == "plus":
            right_main_width = 0.
        elif args.subfigures == "minus":
            left_main_width = 0.

        if args.no_overhangs:
            topplots_hight, rightplots_width = 0., 0.

        gs_outer = gridspec.GridSpec(1, 3, wspace=center, hspace=0., left=left, right=right, top=top, bottom=bottom,
                                     width_ratios=[subfigure_str_width, left_main_width, right_main_width])
        gs_l = gridspec.GridSpecFromSubplotSpec(2, 4, hspace=0., wspace=0., subplot_spec=gs_outer[1], 
                                                width_ratios=[1., rightplots_width, cb_dist, cb_width], 
                                                height_ratios=[topplots_hight, 1.])
        gs_r = gridspec.GridSpecFromSubplotSpec(2, 4, hspace=0., wspace=0., subplot_spec=gs_outer[2], 
                                                width_ratios=[1., rightplots_width, cb_dist, cb_width], 
                                                height_ratios=[topplots_hight, 1.])

        axsub = plt.Subplot(fig, gs_outer[0])
        
        if left_main_width:
            ax1  = plt.Subplot(fig, gs_l[1, 0])
            ax1s = plt.Subplot(fig, gs_l[1, 3])
            if not args.no_overhangs:
                ax1r = plt.Subplot(fig, gs_l[1, 1])
                ax1t = plt.Subplot(fig, gs_l[0, 0])
                ax1i = plt.Subplot(fig, gs_l[0, 1])
            else:
                ax1r, ax1t, ax1i = None, None, None

        
        if right_main_width:
            ax2  = plt.Subplot(fig, gs_r[1, 0])
            ax2s = plt.Subplot(fig, gs_r[1, 3])
            if not args.no_overhangs:
                ax2r = plt.Subplot(fig, gs_r[1, 1])
                ax2t = plt.Subplot(fig, gs_r[0, 0])
                ax2i = plt.Subplot(fig, gs_r[0, 1])
            else:
                ax2r, ax2t, ax2i = None, None, None

         ####### left plot #######

        axsub.get_yaxis().set_visible(False)
        axsub.get_xaxis().set_visible(False)
        axsub.spines['top'].set_visible(False)
        axsub.spines['right'].set_visible(False)
        axsub.spines['bottom'].set_visible(False)
        axsub.spines['left'].set_visible(False)
        if args.subfigure:
            axsub.text(0., 1.5, args.subfigure, horizontalalignment='left', verticalalignment='top', 
                       transform=axsub.transAxes, fontsize=args.fontsize+2)

        displayed_hits = [[],[]]
        collections = []


        ax_systems = []
        axes = [axsub]
        if left_main_width:
            ax_systems.append((1, ax1, ax1r, ax1t, ax1i, ax1s))
            axes.extend([ax1, ax1s])
            if not args.no_overhangs:
                axes.extend([ax1r, ax1t, ax1i])
        if right_main_width:
            ax_systems.append((-1, ax2, ax2r, ax2t, ax2i, ax2s))
            axes.extend([ax2, ax2s])
            if not args.no_overhangs:
                axes.extend([ax2r, ax2t, ax2i])

        for ori, ax, axr, axt, axi, axs in ax_systems:
            subj_starts = []
            subj_ends = []
            quer_starts = []
            quer_tails = []
            colors = []
            colors_starts = []
            colors_tails = []
            num_plotted = {1:0, -1:0}

            for i,hit in enumerate(alignments):

                # hide reads for which no primer sequences were detected
                if args.primers and args.hide:
                    if hit.qid in reads:
                        if args.hide == "all_unclassified":
                            if len(reads[hit.qid].pids_5) != 1 or len(reads[hit.qid].pids_3) != 1:
                                continue
                        elif args.hide == "5'_unclassified":
                            if len(reads[hit.qid].pids_5) != 1:
                                continue
                        elif args.hide == "3'_unclassified":
                            if len(reads[hit.qid].pids_3) != 1:
                                continue
                        elif args.hide == "all_classified":
                            if len(reads[hit.qid].pids_5) == 1 or len(reads[hit.qid].pids_3) == 1:
                                continue
                        elif args.hide == "5'_classified":
                            if len(reads[hit.qid].pids_5) == 1:
                                continue
                        elif args.hide == "3'_classified":
                            if len(reads[hit.qid].pids_3) == 1:
                                continue
                    else:
                        if args.hide == "all_unclassified" or \
                           args.hide == "5'_unclassified" or \
                           args.hide == "3'_unclassified":
                            continue

                if hit.strand == ori:
                    if ori == 1:
                        if (args.xlim[0] <= hit.r_st <= args.xlim[1]) and (args.ylim[0] <= hit.r_en <= args.ylim[1]):
                            subj_starts.append(hit.r_st)
                            subj_ends.append(hit.r_en)
                            displayed_hits[0].append(hit)
                        else:
                            continue
                    if ori == -1:
                        if (args.xlim[0] <= hit.r_en <= args.xlim[1]) and (args.ylim[0] <= hit.r_st <= args.ylim[1]):
                            subj_starts.append(hit.r_en)
                            subj_ends.append(hit.r_st)
                            displayed_hits[1].append(hit)
                        else:
                            continue

                    if hit.joined:
                        colors.append('C1')
                    else:
                        colors.append('black')
                    num_plotted[ori] += 1

                    if args.primer_as_terminal and args.hide == 'all_unclassified':
                        quer_starts.append(hit.q_st - reads[hit.qid].pos_5[0])
                        quer_tails.append((hit.qlen - hit.q_en) - reads[hit.qid].pos_3[0])
                    elif args.primer_as_terminal and args.hide == "5'_unclassified":
                        quer_starts.append(hit.q_st - reads[hit.qid].pos_5[0])
                        quer_tails.append(hit.qlen - hit.q_en)
                    elif args.primer_as_terminal and args.hide == "3'_unclassified":
                        quer_starts.append(hit.q_st)
                        quer_tails.append((hit.qlen - hit.q_en) - reads[hit.qid].pos_3[0])
                    else:
                        quer_starts.append(hit.q_st)
                        quer_tails.append(hit.qlen - hit.q_en)

                    #if args.primers:
                    if args.annotation:
                        if hit.qid not in reads:
                            colors_starts.append(unclassified_color)
                            colors_tails.append(unclassified_color)
                        else:
                            if len(reads[hit.qid].pids_5) == 1:
                                colors_starts.append(primer_colors[reads[hit.qid].pids_5[0]])
                            elif len(reads[hit.qid].pids_5) == 0:
                                colors_starts.append(unclassified_color)
                            else:
                                colors_starts.append(multiple_color)

                            if len(reads[hit.qid].pids_3) == 1:
                                colors_tails.append(primer_colors[reads[hit.qid].pids_3[0]])
                            elif len(reads[hit.qid].pids_3) == 0:
                                colors_tails.append(unclassified_color)
                            else:
                                colors_tails.append(multiple_color)

            ####### get vmin, vmax #######

            xmin, xmax = args.xlim
            ymin, ymax = args.ylim

            axt_ymin = min([0, min(quer_starts)]) if quer_starts else 0
            axt_ymax = max(quer_starts) if quer_starts else args.max_qst
            axr_xmin = min([0, min(quer_tails)]) if quer_starts else 0
            axr_xmax = max(quer_tails) if quer_starts else args.max_qen
            
            if args.hist2d:
                max_bin_counts = []

                bin_edges = [list(range(args.xlim[0], args.xlim[1]+((args.hist2d+1)//2), args.hist2d)), 
                             list(range(args.ylim[0], args.ylim[1]+((args.hist2d+1)//2), args.hist2d))]
                range_ = [[args.xlim[0], args.xlim[1]], [args.ylim[0], args.ylim[1]]]
                bin_counts = get_bin_counts(subj_starts, subj_ends, bin_edges, args.hist2d, range_)
                max_bin_counts.append(bin_counts.max())

                bin_edges = [list(range(int(xmin), int(xmax)+((args.hist2d+1)//2), args.hist2d)), 
                             list(range(axt_ymin, axt_ymax+((args.hist2d+1)//2), args.hist2d))]
                range_ = [[xmin, xmax], [axt_ymin, axt_ymax]]
                bin_counts = get_bin_counts(subj_starts, quer_starts, bin_edges, args.hist2d, range_)
                max_bin_counts.append(bin_counts.max())

                bin_edges = [list(range(axr_xmin, axr_xmax+((args.hist2d+1)//2), args.hist2d)), 
                             list(range(int(ymin), int(ymax)+((args.hist2d+1)//2), args.hist2d))]
                range_ = [[axr_xmin, axr_xmax], [ymin, ymax]]
                bin_counts = get_bin_counts(quer_tails, subj_ends, bin_edges, args.hist2d, range_)
                max_bin_counts.append(bin_counts.max())

                vmax = max(max_bin_counts)


            ####### center plots #######

            ax.plot([0, 1], [0, 1], transform=ax.transAxes, linewidth=0.5, color="black", alpha=0.25)     # diagonal

            if args.hist2d:
                bin_edges = [list(range(args.xlim[0], args.xlim[1]+((args.hist2d+1)//2), args.hist2d)), 
                             list(range(args.ylim[0], args.ylim[1]+((args.hist2d+1)//2), args.hist2d))]
                range_ = [[args.xlim[0], args.xlim[1]], [args.ylim[0], args.ylim[1]]]
                print("bin edges: ", bin_edges[0][0], bin_edges[0][-1], bin_edges[1][0], bin_edges[1][-1])
                if args.log:
                    _,_,_,qm = ax.hist2d(subj_starts, subj_ends, bins=bin_edges, range=range_, cmin=1, norm=LogNorm(), 
                                         cmap=args.colormap, vmax=vmax, linewidth=0., antialiased=True, rasterized=True,
                                         edgecolors='none')
                    im = qm
                else:
                    _,_,_,qm = ax.hist2d(subj_starts, subj_ends, bins=bin_edges, range=range_, cmin=1, 
                                         cmap=args.colormap, vmax=vmax, linewidth=0., antialiased=True, rasterized=True,
                                         edgecolors='none')
                    im = qm
            else:
                if args.interactive:
                    collections.append([])
                    for subj_start, subj_end, color in zip(subj_starts, subj_ends, colors):
                        collection = ax.scatter(subj_start, subj_end, s=args.surface, marker="s", c=color, 
                                                edgecolors="none")
                    
                        fc = collection.get_facecolors()
                        fc[:, -1] = args.alpha
                        collection.set_facecolors(fc)

                        collections[-1].append(collection)
                else:
                    ax.scatter(subj_starts, subj_ends, s=args.surface, marker="s", c=colors, edgecolors="none", 
                               alpha=args.alpha)


            if ori == 1:
                ax.set_xlabel("subject start")
                ax.set_ylabel("subject end")
            else:
                ax.set_xlabel("subject end")
                ax.set_ylabel("subject start")
            ax.set_xlim(args.xlim[0], args.xlim[1])
            ax.set_ylim(args.ylim[0], args.ylim[1])

            if not args.no_overhangs:
                ####### top plots #######
                
                if args.hist2d:
                    bin_edges = [list(range(int(xmin), int(xmax)+((args.hist2d+1)//2), args.hist2d)), 
                                 list(range(axt_ymin, axt_ymax+((args.hist2d+1)//2), args.hist2d))]
                    range_ = [[xmin, xmax], [axt_ymin, axt_ymax]]
                    if args.log:
                        _,_,_,qm = axt.hist2d(subj_starts, quer_starts, bins=bin_edges, range=range_, cmin=1, 
                                              norm=LogNorm(), cmap=args.colormap, vmax=vmax, linewidth=0., 
                                              antialiased=True, rasterized=True, edgecolors='none')
                    else:
                        _,_,_,qm = axt.hist2d(subj_starts, quer_starts, bins=bin_edges, range=range_, cmin=1, 
                                              cmap=args.colormap, vmax=vmax, linewidth=0., antialiased=True, 
                                              rasterized=True, edgecolors='none')
                else:
                    c_ = colors_starts if args.primers else "black"
                    axt.scatter(subj_starts, quer_starts, s=args.surface, marker="s", edgecolors="none", c=c_, 
                                alpha=args.alpha)

                axt.set_xlim(xmin, xmax)
                axt.set_ylim(axt_ymin, axt_ymax)
                if args.primer_as_terminal and (args.hide == 'all_unclassified' or args.hide == "5'_unclassified"):
                    axt.set_ylabel("5' primer dist")
                else:
                    axt.set_ylabel("5' overhang")
                axt.get_xaxis().set_visible(False)
                axt.spines['top'].set_visible(False)
                axt.spines['right'].set_visible(False)
                if ori == 1:
                    axt.set_title("plus alignments (n = {})".format(num_plotted[ori]))
                else:
                    axt.set_title("minus alignments (n = {})".format(num_plotted[ori]))

                ####### right plots #######

                if args.hist2d:
                    bin_edges = [list(range(axr_xmin, axr_xmax+((args.hist2d+1)//2), args.hist2d)), 
                                 list(range(int(ymin), int(ymax)+((args.hist2d+1)//2), args.hist2d))]
                    range_ = [[axr_xmin, axr_xmax], [ymin, ymax]]
                    if args.log:
                        _,_,_,qm = axr.hist2d(quer_tails, subj_ends, bins=bin_edges, range=range_, cmin=1, 
                                              norm=LogNorm(), cmap=args.colormap, vmax=vmax, linewidth=0., 
                                              antialiased=True, rasterized=True, edgecolors='none')
                    else:
                        _,_,_,qm = axr.hist2d(quer_tails, subj_ends, bins=bin_edges, range=range_, cmin=1, 
                                              cmap=args.colormap, vmax=vmax, linewidth=0., antialiased=True, 
                                              rasterized=True, edgecolors='none')
                else:
                    c_ = colors_tails if args.primers else "black"
                    axr.scatter(quer_tails, subj_ends, s=args.surface, marker="s", edgecolors="none", c=c_, 
                                alpha=args.alpha)
                axr.set_ylim(ymin, ymax)
                axr.set_xlim(axr_xmin, axr_xmax)
                if args.primer_as_terminal and (args.hide == 'all_unclassified' or args.hide == "3'_unclassified"):
                    axr.set_xlabel("3' primer dist")
                else:
                    axr.set_xlabel("3' overhang")
                axr.get_yaxis().set_visible(False)
                axr.spines['top'].set_visible(False)
                axr.spines['right'].set_visible(False)

                ####### info plots #######

                if args.primers:
                    custom_lines = [Line2D([0], [0], marker='o',color='w', markerfacecolor=unclassified_color),
                                    Line2D([0], [0], marker='o',color='w', markerfacecolor=multiple_color)]
                    custom_lines += [Line2D([0], [0], marker='o',color='w', markerfacecolor=primer_colors[k])\
                                    for k in primer_ids]
                    axi.legend(custom_lines, ["unclassified", "multiple"] + primer_ids)
                axi.get_yaxis().set_visible(False)
                axi.get_xaxis().set_visible(False)
                axi.spines['top'].set_visible(False)
                axi.spines['right'].set_visible(False)
                axi.spines['bottom'].set_visible(False)
                axi.spines['left'].set_visible(False)

            ####### scala plots #######

            if args.hist2d:
                cb = mpl.colorbar.Colorbar(mappable=im, ax=axs)
                ticks = cb.get_ticks()
                tick_labels = [str(tick) if tick % 1 else str(int(tick)) for tick in ticks]
                cb.set_ticks(ticks)
                cb.set_ticklabels(tick_labels)
                print(ticks)
                cb.set_label('reads per bin')
            else:
                vmax = int(1. / args.alpha)
                cb = mpl.colorbar.ColorbarBase(axs, cmap=cm.binary, norm=mpl.colors.Normalize(vmin=0, vmax=vmax))
                ticks = cb.get_ticks()
                tick_labels = [str(tick) if tick % 1 else str(int(tick)) for tick in ticks[:-1]] 
                if ticks[-1] % 1:
                    tick_labels += ["≥{}".format(ticks[-1])]
                else:
                    tick_labels += ["≥{}".format(int(ticks[-1]))]
                cb.set_ticks(ticks)
                cb.set_ticklabels(tick_labels)
                cb.set_label('stacked reads')


        for ax in axes:
            ax.tick_params(width=args.spines_linewidth)
            fig.add_subplot(ax)

        print(" done")

        if args.out:
            if "{}" not in args.out:
                l = args.out.split(".")
                l.insert(-1, "_{}")
                args.out = ".".join(l)

            plt.savefig(args.out.format(subject))
        else:
            if args.interactive:
                selections = [[]]
                patches = []
                selector = SelectFromCollection(ax1, collections[0])

                def key_press_event(event):
                    if event.key == "enter":
                        print("Selected points:")
                        selected_hits = []
                        colors_ = []
                        for i in range(len(selector.collections)):
                            coordinates = selector.xys[i][selector.ind[i]]
                            if len(coordinates):
                                selected_hits.append(displayed_hits[0][i])
                                colors_.append(selector.fc[i][selector.ind[i]][0])
                        path = selector.accept()

                        patch = mpatches.PathPatch(path, facecolor='None', edgecolor='k', lw=args.spines_linewidth)
                        patches.append(patch)
                        ax1.add_patch(patch)
                        fig.canvas.draw()

                        # linear regression through 5' overhangs
                        X = [hit.r_st for hit in selected_hits]
                        Y = [hit.q_st for hit in selected_hits]
                        m,b = np.polyfit(X, Y, 1)
                        poly1d_fn = np.poly1d((m,b))
                        ax1t.plot([0, max(X)],poly1d_fn([0, max(X)]), ':k', linewidth=args.spines_linewidth)
                        print("5' estimate: ", ((0.-b)/m))

                        # linear regression through 3' overhangs
                        X = [hit.qlen - hit.q_en for hit in selected_hits]
                        Y = [hit.r_en for hit in selected_hits]
                        m,b = np.polyfit(X, Y, 1)
                        poly1d_fn = np.poly1d((m,b))
                        ax1r.plot([0, max(X)],poly1d_fn([0, max(X)]), ':k', linewidth=args.spines_linewidth)
                        print("3' estimate: ", b)                        

                        fig.canvas.draw()
                    elif event.key == "escape":
                        selector.disconnect()
                        fig.canvas.draw()
                    elif event.key == "shift":
                        selector.shift = True
                    elif event.key == "l":
                        selector.selector.disconnect_events()
                        selector.selector = LassoSelector(ax1, onselect=selector.onselect_lasso)
                        ax1.set_yscale("linear")
                    elif event.key == "e":
                        selector.selector.disconnect_events()
                        selector.selector = EllipseSelector(ax1, onselect=selector.onselect_ellipse, drawtype='line')
                        ax1.set_yscale("linear")

                def key_release_event(event):
                    if event.key == "shift":
                        selector.shift = False

                fig.canvas.mpl_connect("key_press_event", key_press_event)
                fig.canvas.mpl_connect("key_release_event", key_release_event)

            plt.show()

def preprocess_data(paf_content, subjects, circular, max_dist, multiple, min_qlen, max_qst, max_qen, project):
    """
        Parses a .paf file and returns a subset of its contents as a list, selected based on the
        user settings.

        Parameters
        ----------
        paf_fn: str
            Path to the .paf file.
        subjects: [str]
            List of the subject ids for which a plot shall be generated.
            
        Returns
        -------
        dict
            A dictionary containing the subject ids as keys and lists of alignments
            as values.
    """

    alignments = {subject:[] for subject in subjects}
    for qid,subj in paf_content:
        # remove alignments of reads that are too short
        if min_qlen:
            if paf_content[qid,subj][0].qlen < min_qlen:
                continue
        # remove alignments against subjects that were not selected
        if subj in alignments:
            alignments[subj].append(paf_content[qid,subj])
        elif not subjects:
            alignments[subj] = [paf_content[qid,subj]]
    
    if circular:
        alignments = circularize(alignments, max_dist)

    if multiple == "all":
        for subj in alignments:
            alignments[subj] = [hit for hits in alignments[subj] for hit in hits]
    elif multiple == "none":
        for subj in alignments:
            alignments[subj] = [hit for hits in alignments[subj] for hit in hits if len(hits) == 1]
    elif multiple == "joined":
        for subj in alignments:
            new_hits = []
            for hits in alignments[subj]:
                if len(hits) == 1:
                    new_hits.append(hits[0])
                    continue

                ori = {hit.strand for hit in hits}
                if not len(ori) == 1:
                    continue # exclude reads that map (partly) in both orientations
                ori = ori.pop()
                sorted_by_qst = copy.deepcopy(hits)
                sorted_by_rst = copy.deepcopy(hits)
                sorted_by_qst.sort(key=lambda hit: hit.q_st)
                sorted_by_rst.sort(key=lambda hit: hit.r_st)

                # exclude reads that map to the reference in a non coherent order
                if ori == -1:
                    sorted_by_rst.reverse()
                if circular:
                    for i in range(len(hits)):
                        if sorted_by_qst == sorted_by_rst:
                            break
                        hit = sorted_by_rst.pop(0)
                        sorted_by_rst.append(hit)
                    else:
                        continue
                else:
                    if not sorted_by_qst == sorted_by_rst:
                        continue 

                hit_l = sorted_by_qst[0]
                hit_r = sorted_by_qst[-1]
                if ori == 1:
                    r_st = hit_l.r_st
                    r_en = hit_r.r_en
                else:
                    r_st = hit_r.r_st
                    r_en = hit_l.r_en
                new_hit = PAFcontent(qid=hit_l.qid, qlen=hit_l.qlen, q_st=hit_l.q_st, q_en=hit_r.q_en, strand=ori, 
                                     ctg=subj, ctg_len=hit_l.ctg_len, r_st=r_st, r_en=r_en, mlen=None, 
                                     blen=sum([hit.blen for hit in sorted_by_qst]), 
                                     mapq=min([hit.mapq for hit in sorted_by_qst]), 
                                     cigar_str=None, others=sorted_by_qst[:], joined=True, 
                                     circularized=any([hit.circularized for hit in sorted_by_qst]))
                new_hits.append(new_hit)
            alignments[subj] = new_hits
    elif multiple == "best":
        for subj in alignments:
            alignments[subj] = [max(hits, key=lambda hit: hit.mapq) for hits in alignments[subj]]
    elif multiple == "longest":
        for subj in alignments:
            alignments[subj] = [max(hits, key=lambda hit: hit.blen) for hits in alignments[subj]]
    elif multiple == "close_to_q_st":
        for subj in alignments:
            alignments[subj] = [min(hits, key=lambda hit: hit.q_st) for hits in alignments[subj]]
    elif multiple == "close_to_q_en":
        for subj in alignments:
            alignments[subj] = [max(hits, key=lambda hit: hit.q_en) for hits in alignments[subj]]

    if max_qst:
        for subj in alignments:
            alignments[subj] = [hit for hit in alignments[subj] if hit.q_st <= max_qst]

    if max_qen:
        for subj in alignments:
            alignments[subj] = [hit for hit in alignments[subj] if (hit.qlen - hit.q_en) <= max_qen]

    if project:
        projected_alignments = {}
        for subj in alignments:
            projected_alignments[subj] = []
            for hit in alignments[subj]:
                if hit.strand == 1:
                    new_hit = PAFcontent(qid=hit.qid, qlen=hit.qlen, q_st=0, q_en=hit.qlen, strand=hit.strand, 
                                         ctg=hit.ctg, ctg_len=hit.ctg_len, r_st=hit.r_st - hit.q_st, 
                                         r_en=hit.r_en + (hit.qlen - hit.q_en), mlen=hit.mlen, blen=hit.blen, 
                                         mapq=hit.mapq, cigar_str=hit.cigar_str, others=hit.others, 
                                         joined=hit.joined, circularized=hit.circularized)
                elif hit.strand == -1:
                    new_hit = PAFcontent(qid=hit.qid, qlen=hit.qlen, q_st=0, q_en=hit.qlen, strand=hit.strand, 
                                         ctg=hit.ctg, ctg_len=hit.ctg_len, r_st=hit.r_st - (hit.qlen - hit.q_en), 
                                         r_en=hit.r_en + hit.q_st, mlen=hit.mlen, blen=hit.blen, mapq=hit.mapq, 
                                         cigar_str=hit.cigar_str, others=hit.others, 
                                         joined=hit.joined, circularized=hit.circularized)
                projected_alignments[subj].append(new_hit)
        alignments = projected_alignments

    return alignments

def parse_paf(fn):
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
            hit = PAFcontent( *[int_or_str(s) for s in line.split()[:12]], cigar_str, None, False, False)
            if (hit.qid,hit.ctg) in alignments:
                alignments[(hit.qid,hit.ctg)].append(hit)
            else:
                alignments[(hit.qid,hit.ctg)] = [hit]
    return alignments

def keep_best_alignment(alignments):
    best_per_read = {}
    for qid,ctg in alignments:
        hits = alignments[(qid,ctg)]
        best_hit = hits[0]
        for hit in hits:
            if hit.mapq > best_hit.mapq:
                best_hit = hit
        best_per_read[qid] = best_hit
    return {(qid,best_per_read[qid].ctg) : best_per_read[qid] for qid in best_per_read}

def circularize(alignments, max_dist):
    new_alignments = {subject:[] for subject in alignments}
    for subject in alignments:
        for hits in alignments[subject]:
        #hits = alignments[(qid,ctg)]
            for ori in [1, -1]:
                ori_hits = [hit for hit in hits if hit.strand == ori]
                if len(ori_hits) <= 1:
                    continue
                ori_hits.sort(key=lambda x: x.r_st)
                hit_l = ori_hits[0]
                hit_r = ori_hits[-1]

                if ori == 1 and not (hit_r.q_en < hit_l.q_st):
                    continue
                elif ori == -1 and not (hit_l.q_en < hit_r.q_st):
                    continue
                # hit_l and hit_r are two non-overlapping alignments of the same 
                # read mapping to the reference in the same orientation

                if ori == 1:
                    if 0 <= dist(hit_l.r_st, hit_r.r_en, hit_l.ctg_len, True) <= max_dist and \
                       hit_l.q_st - hit_r.q_en <= max_dist:
                        hits.remove(hit_l)
                        hits.remove(hit_r)
                        new_hit = PAFcontent(qid=hit_l.qid, qlen=hit_l.qlen, q_st=hit_r.q_st, q_en=hit_l.q_en, 
                                             strand=ori, ctg=hit_l.ctg, ctg_len=hit_l.ctg_len, r_st=hit_r.r_st, 
                                             r_en=hit_l.r_en, mlen=None, blen=hit_r.blen+hit_l.blen, 
                                             mapq=min([hit_l.mapq, hit_r.mapq]), cigar_str=None, others=[hit_l, hit_r], 
                                             joined=False, circularized=True)
                        hits.append(new_hit)
                elif ori == -1:
                    if 0 <= dist(hit_l.r_st, hit_r.r_en, hit_l.ctg_len, True) <= max_dist and \
                       hit_r.q_st - hit_l.q_en <= max_dist:
                        hits.remove(hit_l)
                        hits.remove(hit_r)
                        new_hit = PAFcontent(qid=hit_l.qid, qlen=hit_l.qlen, q_st=hit_l.q_st, q_en=hit_r.q_en, 
                                             strand=ori, ctg=hit_l.ctg, ctg_len=hit_l.ctg_len, r_st=hit_r.r_st, 
                                             r_en=hit_l.r_en, mlen=None, blen=hit_r.blen+hit_l.blen, 
                                             mapq=min([hit_l.mapq, hit_r.mapq]), cigar_str=None, others=[hit_l, hit_r], 
                                             joined=False, circularized=True)
                        hits.append(new_hit)
            new_alignments[subject].append(hits)
    return new_alignments

def parse_annotation_file(fn):
    reads = {}
    with open(fn, "r") as f:
        for line in f.readlines():
            line = line.rstrip("\n")
            rid, pids_5, pos_5, pids_3, pos_3 = line.split("\t")
            pids_5 = [i for i in pids_5.split(" ") if i]
            pos_5 = [int(i) for i in pos_5.split(" ") if i]
            pids_3 = [i for i in pids_3.split(" ") if i]
            pos_3 = [int(i) for i in pos_3.split(" ") if i]
            reads[rid] = Read(id=rid, len=None, tlen=None, seq_5=None, seq_3_rc=None, 
                              pids_5=pids_5, pos_5=pos_5, pids_3=pids_3, pos_3=pos_3)
    return reads

def get_bin_counts(X, Y, bin_edges, bin_width, range_):
    bin_counts = [[0 for i in range(len(bin_edges[1]))] for j in range(len(bin_edges[0]))]
    for x,y in zip(X, Y):
        if range_[0][0] <= x < range_[0][1] and range_[1][0] <= y < range_[1][1]:
            x_ = x - range_[0][0]
            y_ = y - range_[1][0]
            bin_x = x_ // bin_width
            bin_y = y_ // bin_width
            bin_counts[bin_x][bin_y] += 1
    return np.array(bin_counts)

def dist(a,b,subj_len, circular):
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
        if circular:
            if (a-b) < ((subj_len+b) - a):
                return a - b
            else:
                return a - (subj_len+b)
        else:
            return a - b
    elif a < b:
        if circular:
            if (b-a) < ((subj_len+a) - b):
                return a - b
            else:
                return (subj_len+a) - b
        else:
            return a - b
    else:
        return 0

if __name__ == '__main__':
    args = parse_args()
    main()
