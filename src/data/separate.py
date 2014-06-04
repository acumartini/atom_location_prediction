with open( "test.tsv", "r" ) as train:
	idx = 0
	for line in train:
		print line
		with open( "./test/" + str(idx) + ".tsv", "w" ) as instance:
			instance.write( line )
		idx += 1