from flytracker.analysis.parsing import combine_results

# Parsing old csv into single hdf file
combine_results('data/testing_data/4arenas/old_results/', 
                 'data/testing_data/4arenas/old_combined.hdf')