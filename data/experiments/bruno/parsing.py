from flytracker.analysis.parsing import combine_results

# Parsing old csv into single hdf file
combine_results('old_results/', 
                 'old_combined.hdf')