

from model import SurfaceDistanceUsingFmap




def main():
    # read the files
    ...
    
    # 
    for surf1, surf2 in molecular_pairs:

        sd = SurfaceDistanceUsingFmap()
        sd.compute_Fmaps_n_distance(surf1, surf2)
        
        
