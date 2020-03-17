#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
@author: David Brandt (dbrandt@cebitec.uni-bielefeld.de)
https://github.com/MarkusHaak/Radukic_Brandt_2020

Python 2.7 script to parse blastn output (-outfmt 6) from stdin and assign
highest scoring blastn hit (bitscore) to respective subject sequence

This file is part of the repository Radukic_Brandt_2020. The scripts in Radukic_Brandt_2020 are free software: 
You can redistribute it and/or modify them under the terms of the GNU General Public License as published by the 
Free Software Foundation, either version 3 of the License, or (at your option) any later version. Radukic_Brandt_2020
scripts are distributed in the hope that they will be useful, but WITHOUT ANY WARRANTY; without even the implied 
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
details. You should have received a copy of the GNU General Public License along with mega. If not, 
see <http://www.gnu.org/licenses/>.
"""

import sys
import re

#initialization of counter variables for BLASTn hit assignment

aav_counter = 0
pzmb0088_counter = 0
pzmb0504_counter = 0
pzmb0347_counter = 0
hg_counter = 0

read_counter = 0
short_counter = 0

#lists collecting subject_id, alignment length and bitscore of high-scoring segment pairs (HSPs)

subject_list = [1]
length_list = [1]
bitscore_list = [0]

current_query = "0"

for line in sys.stdin:
	
	elements = line.split("	")
	
	#skip short alignments
	
	if int(elements[3]) < 250:
		if current_query != elements[0]:
			short_counter +=1
			read_counter += 1
		else:
			pass
	
	#collect multiple HSPs of a single query in subject_id, alignment length and bitscore lists
			
	elif current_query == elements[0]:
		subject_list.append(elements[1])
		length_list.append(elements[3])
		bitscore_list.append(float(elements[11].strip()))
	
	#when reaching the next query, assign previous query to a subject sequence
		
	elif current_query != elements[0]:
		if read_counter == 0:
			subject_list.append(elements[1])
			length_list.append(elements[3])
			bitscore_list.append(float(elements[11].strip()))
		read_counter += 1
		
		#determine best BLASTn HSP via highest bit-score
		
		index_of_highest_bitscore = bitscore_list.index(max(bitscore_list))
		
		if subject_list[index_of_highest_bitscore] == "AAV":
			aav_counter += 1
		elif subject_list[index_of_highest_bitscore] == "pZMB0088":
			pzmb0088_counter += 1
		elif subject_list[index_of_highest_bitscore] == "pZMB0504":
			pzmb0504_counter += 1		
		elif subject_list[index_of_highest_bitscore] == "pZMB0347":
			pzmb0347_counter += 1
		elif re.search(r"chr.*",subject_list[index_of_highest_bitscore]):
			hg_counter += 1

		#reset the lists when the loop reaches the next query
		
		subject_list = []
		length_list = []
		bitscore_list = []

		subject_list.append(elements[1])
		length_list.append(elements[3])
		bitscore_list.append(float(elements[11].strip()))
		
	current_query = elements[0]

#output final results of hit assignment to stdout

print "Total: %d reads" %(read_counter)
print "AAV: %d reads (%.2f percent) (Alignment length: > 250)" %(aav_counter,round((float(aav_counter)/float(read_counter-short_counter)*100),2))
print "pZMB0088: %d reads (%.2f percent) (Alignment length: > 250)" %(pzmb0088_counter,round((float(pzmb0088_counter)/float(read_counter-short_counter)*100),2))
print "pZMB0504: %d reads (%.2f percent) (Alignment length: > 250)" %(pzmb0504_counter,round((float(pzmb0504_counter)/float(read_counter-short_counter)*100),2))
print "pZMB0347: %d reads (%.2f percent) (Alignment length: > 250)" %(pzmb0347_counter,round((float(pzmb0347_counter)/float(read_counter-short_counter)*100),2))
print "hg38: %d reads (%.2f percent) (Alignment length: > 250)" %(hg_counter,round((float(hg_counter)/float(read_counter-short_counter)*100),2))
print "too short: %d reads (Alignment length: < 250)" %(short_counter)
