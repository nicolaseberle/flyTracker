from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix


class Hungarian:
    def __call__(self, coordinates_i, coordinates_j):
        identities_j = linear_sum_assignment(distance_matrix(coordinates_i, coordinates_j))[1].squeeze()
        return identities_j
    
    
