package nl.uva.imcn.algorithms;

import java.io.*;
import java.util.*;


import nl.uva.imcn.structures.*;
import nl.uva.imcn.libraries.*;
import nl.uva.imcn.utilities.*;
import nl.uva.imcn.methods.*;

/**
 *
 *  This algorithm handles Fuzzy C-Means operations
 *  for 3D data as in  in FANTASM (membership, centroid computations)
 *
 *	@version    November 2019
 *	@author     Pierre-Louis Bazin
 *		
 *
 */
 
public class FuzzyCmeans {
		
	// numerical quantities
	private static final	float   INF=1e30f;
	private static final	float   ZERO=1e-30f;
	
	// data buffers
	private 	float[]			image;  			// original image
	private 	float[][]		mems;				// membership function
	private 	float[]			centroids;			// cluster centroids
	private 	boolean[]		mask;   			// image mask: true for data points
	private     int[]           id;                 // re-ordering of data by increasing centroid values
	private static	int 		nx,ny,nz, nxyz;     // images dimensions
	private static	float 		rx,ry,rz;   		// images resolutions
	
	// parameters
	private 	int 		clusters;    // number of clusters in original membership: > clusters if outliers
	private 	float 		smoothing;	// MRF smoothing
 	private		float		fuzziness = 2.0f;	// fuzziness factor
	private		PowerTable	power, invpower;
	private		float		imgvar = 1.0f;		// image variance
	private     float       maxDist = 0.01f;
	private     int         maxIter = 50;
	
			
	// computation variables
	private		float[]		prev;	// previous membership values
	
	// for debug and display
	static final boolean		debug=true;
	static final boolean		verbose=true;
	
	public final void setImage(float[] val) { image = val; }
	public final void setMaskImage(int[] val) { 
	    mask = new boolean[nxyz];
	    for (int xyz=0;xyz<nxyz;xyz++) mask[xyz] = (val[xyz]>0); 
	}
	public final void initZeroMaskImage() { 
	    mask = new boolean[nxyz];
	    for (int xyz=0;xyz<nxyz;xyz++) mask[xyz] = (image[xyz]!=0); 
	}
	
	public final void setClusterNumber(int val) { clusters = val; }
	public final void setSmoothing(float val) { smoothing = val; }
	public final void setFuzziness(float val) { fuzziness = val; }
	public final void setMaxDist(float val) { maxDist = val; }
	public final void setMaxIter(int val) { maxIter = val; }
	
	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
		
	// to be used for JIST definitions, generic info / help
	public final String getPackage() { return "IMCN Toolkit"; }
	public final String getCategory() { return "Clustering"; }
	public final String getLabel() { return "Fuzzy C-means"; }
	public final String getName() { return "FuzzyCmeans"; }

	public final String[] getAlgorithmAuthors() { return new String[]{"Pierre-Louis Bazin"}; }
	public final String getAffiliation() { return "Integrated Model-based Cognitive Neuroscience Reseaerch Unit, Universiteit van Amsterdam"; }
	public final String getDescription() { return "Fuzzy C-means algorithm adapted from (Pham, 2001)"; }
	public final String getLongDescription() { return getDescription(); }
		
	public final String getVersion() { return "1.0"; };

	// create outputs
	public final float[] getMembership(int n) { return mems[id[n+1]]; }
   
    public final void execute() {		
		
        // image range
        float Imin = INF, Imax = -INF;
        for (int xyz=0;xyz<nxyz;xyz++) {
            if (image[xyz]<Imin) Imin = image[xyz];
            if (image[xyz]>Imax) Imax = image[xyz];
        }   
		
        // init all the new arrays
        mems = new float[clusters][nxyz];
        centroids = new float[clusters];
        prev = new float[clusters];
        if (fuzziness!=2) {
            power = new PowerTable(0.0f , 1.0f , 0.000001f , fuzziness );
            invpower = new PowerTable(0, (Imax-Imin)*(Imax-Imin), 0.000001f*(Imax-Imin)*(Imax-Imin), 1.0f/(1.0f-fuzziness) );
        } else {
            power = null;
            invpower = null;
        }

		// init values
		for (int k=0;k<clusters;k++) for (int xyz=0;xyz<nxyz;xyz++) {
            mems[k][xyz] = 0.0f;
		}
		for (int k=0;k<clusters;k++) {
			centroids[k] = 0.0f;
		}
		
		// init centroids from data range
		centroids[0] = Imin + 0.5f*(Imax-Imin)/(float)clusters;
		for (int k=1;k<clusters;k++)
            centroids[k] = centroids[k-1] + (Imax-Imin)/(float)clusters;

        // initialize the memberships
        computeMemberships();
		
		// main iterations: compute the classes on the image
		
		// with inhomogeneity correction, two steps: first iterates until mild convergence without inhomogeneity correction
		float distance = 0.0f;
		int Niterations = 1;
		boolean stop = false;
		if (Niterations >= maxIter) stop = true;
		while (!stop) {
			if (verbose) System.out.println("iteration " + Niterations + " (max: " + distance + ")\n");
			
			// update centroids
			computeCentroids();
			
			// update membership
			distance = computeMemberships();
			
			// check for segmentation convergence 
			Niterations++;
			if (Niterations > maxIter) stop = true;
			if (distance < maxDist) stop = true;            
		}

		// order the classes in increasing order
		id = computeCentroidOrder();
        
		// generate classification map
		
		
    }
    
    /** 
	 *  compute the FCM membership functions given the centroids
	 *	with the different options (outliers, field, edges, MRF)
	 */
    final public float computeMemberships() {
        
        if (fuzziness!=2) return computeGeneralMemberships();
		
        float distance,dist;
        float den,num;
        float neighbors, ngb;
        
        distance = 0.0f;

        for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
            int xyz = x+nx*y+nx*ny*z;
			if ( mask[xyz] ) {
				den = 0;
				// remember the previous values
				for (int k=0;k<clusters;k++) 
					prev[k] = mems[k][xyz];

				for (int k=0;k<clusters;k++) {
					
					// data term
					num = (image[xyz]-centroids[k])*(image[xyz]-centroids[k]);
					
					// spatial smoothing
					if (smoothing > 0.0f) { 
						ngb = 0.0f;  
						neighbors = 0.0f;
						// case by case	: X+
						if (mask[xyz+1]) for (int m=0;m<clusters;m++) if (m!=k) {
							ngb += mems[m][xyz+1]*mems[m][xyz+1];
							neighbors ++;
						}
						// case by case	: X-
						if (mask[xyz-1]) for (int m=0;m<clusters;m++) if (m!=k) {
							ngb += mems[m][xyz-1]*mems[m][xyz-1];
							neighbors ++;
						}
						// case by case	: Y+
						if (mask[xyz+nx]) for (int m=0;m<clusters;m++) if (m!=k) {
							ngb += mems[m][xyz+nx]*mems[m][xyz+nx];
							neighbors ++;
						}
						// case by case	: Y-
						if (mask[xyz-nx]) for (int m=0;m<clusters;m++) if (m!=k) {
							ngb += mems[m][xyz-nx]*mems[m][xyz-nx];
							neighbors ++;
						}
						// case by case	: Z+
						if (mask[xyz+nx*ny]) for (int m=0;m<clusters;m++) if (m!=k) {
							ngb += mems[m][xyz+nx*ny]*mems[m][xyz+nx*ny];
							neighbors ++;
						}
						// case by case	: Z-
						if (mask[xyz-nx*ny]) for (int m=0;m<clusters;m++) if (m!=k) {
							ngb += mems[m][xyz-nx*ny]*mems[m][xyz-nx*ny];
							neighbors ++;
						}
						if (neighbors>0.0) num = num + smoothing*ngb/neighbors;
					}
					// invert the result
					if (num>ZERO) num = 1.0f/num;
					else num = INF;

					//mems[k][x][y][z] = num;
					mems[k][xyz] = num;
					den += num;
				}

				// normalization
				for (int k=0;k<clusters;k++) {
					mems[k][xyz] = mems[k][xyz]/den;

                    // compute the maximum distance
                    dist = Math.abs(mems[k][xyz]-prev[k]);
                    if (dist > distance) distance = dist;
                }
			} else {
				for (int k=0;k<clusters;k++) 
					mems[k][xyz] = 0.0f;
			}
		}
        return distance;
    } // computeMemberships
    
    /** 
	 *  compute the FCM membership functions given the centroids
	 *	with more complex fuzziness values
	 */
    final public float computeGeneralMemberships() {
        float distance,dist;
        float den,num;
        float neighbors, ngb;
        
        distance = 0.0f;
		for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
			int xyz = x+nx*y+nx*ny*z;
			if ( mask[xyz] ) {
				den = 0;
				// remember the previous values
				for (int k=0;k<clusters;k++) 
					prev[k] = mems[k][xyz];

				for (int k=0;k<clusters;k++) {
					
					// data term
					num = (image[xyz]-centroids[k])*(image[xyz]-centroids[k]);
					
					// spatial smoothing
					if (smoothing > 0.0f) { 
						ngb = 0.0f;  
						neighbors = 0.0f;
						// case by case	: X+
						if (mask[xyz+1]) for (int m=0;m<clusters;m++) if (m!=k) {
							ngb += power.lookup(mems[m][xyz+1],fuzziness);
							neighbors ++;
						}
						// case by case	: X-
						if (mask[xyz-1]) for (int m=0;m<clusters;m++) if (m!=k) {
							ngb += power.lookup(mems[m][xyz-1],fuzziness);
							neighbors ++;
						}
						// case by case	: Y+
						if (mask[xyz+nx]) for (int m=0;m<clusters;m++) if (m!=k) {
							ngb += power.lookup(mems[m][xyz+nx],fuzziness);
							neighbors ++;
						}
						// case by case	: Y-
						if (mask[xyz-nx]) for (int m=0;m<clusters;m++) if (m!=k) {
							ngb += power.lookup(mems[m][xyz-nx],fuzziness);
							neighbors ++;
						}
						// case by case	: Z+
						if (mask[xyz+nx*ny]) for (int m=0;m<clusters;m++) if (m!=k) {
							ngb += power.lookup(mems[m][xyz+nx*ny],fuzziness);
							neighbors ++;
						}
						// case by case	: Z-
						if (mask[xyz-nx*ny]) for (int m=0;m<clusters;m++) if (m!=k) {
							ngb += power.lookup(mems[m][xyz-nx*ny],fuzziness);
							neighbors ++;
						}
						if (neighbors>0.0) num = num + smoothing*ngb/neighbors;
					}
					// invert the result
					if (num>ZERO) num = (float)invpower.lookup(num,1.0f/(1.0f-fuzziness) );
					else num = INF;

					mems[k][xyz] = num;
					den += num;
				}

				// normalization
				for (int k=0;k<clusters;k++) {
					mems[k][xyz] = mems[k][xyz]/den;

                    // compute the maximum distance
                    dist = Math.abs(mems[k][xyz]-prev[k]);
                    if (dist > distance) distance = dist;
                }
			} else {
				for (int k=0;k<clusters;k++) 
					mems[k][xyz] = 0.0f;
			}
		}
        return distance;
    } // computeGeneralMemberships
    
    /**
	 * compute the centroids given the membership functions
	 */
    final public void computeCentroids() {
        float num,den;
		
		if (fuzziness!=2) {
			computeGeneralCentroids();
			return;
		}
        
		for (int k=0;k<clusters;k++) {
            num = 0;
            den = 0;
            for (int xyz=0;xyz<nxyz;xyz++) {
                if (mask[xyz]) {
                    num += mems[k][xyz]*mems[k][xyz]*image[xyz];
                    den += mems[k][xyz]*mems[k][xyz];
                }
            }
            if (den>0.0) {
                centroids[k] = num/den;
            } else {
                centroids[k] = 0.0f;
			}				
        }
        if (verbose) {
			System.out.print("centroids: ("+centroids[0]);
			for (int k=1;k<clusters;k++) System.out.print(", "+centroids[k]);
			System.out.print(")\n");
		}   
        return;
    } // computeCentroids
       
    /**
	 * compute the centroids given the membership functions
	 * for any value of the fuzziness
	 */
    final public void computeGeneralCentroids() {
        float num,den;
		float val;
		
		for (int k=0;k<clusters;k++) {
            num = 0;
            den = 0;
            for (int xyz=0;xyz<nxyz;xyz++) {
				if (mask[xyz]) {
					val = (float)power.lookup(mems[k][xyz],fuzziness);
					num += val*image[xyz];
					den += val;
				}
            }
            if (den>0.0) {
                centroids[k] = num/den;
            } else {
                centroids[k] = 0.0f;
            }
        }
        if (verbose) {
			System.out.print("centroids: ("+centroids[0]);
			for (int k=1;k<clusters;k++) System.out.print(", "+centroids[k]);
			System.out.print(")\n");
		}   
        return;
    } // computeCentroids
    
	
	/** 
	 *	create ids for the centroids based on ordering
	 */
	public final int[] computeCentroidOrder() {
		int[]	id = new int[clusters+1];
		int		lowest;
		float[] cent = new float[clusters];
		
		// copy the centroids
        for (int k=0;k<clusters;k++) 
			cent[k] = centroids[k];
		
		// add the zero
		id[0] = 0;

		// order them from smallest to largest
		for (int k=0;k<clusters;k++) {
			lowest = 0;
			for (int l=1;l<clusters;l++) {
				if (cent[l] < cent[lowest]) {
					lowest = l;
				}
			}
			id[k+1] = lowest+1;
			cent[lowest] = INF;
		}
		// keep order for other class types (outliers, etc)
		for (int k=clusters;k<clusters;k++)
			id[k+1] = k+1;
		
        if (debug) {
			System.out.print("ordering: ("+id[0]);
			for (int k=0;k<clusters;k++)System.out.print(", "+id[k+1]);
			System.out.print(")\n");
		}   
 		return id;
	} // computeCentroidOrder

}
