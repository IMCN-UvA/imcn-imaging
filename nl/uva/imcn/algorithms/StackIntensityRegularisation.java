package nl.uva.imcn.algorithms;

import nl.uva.imcn.utilities.*;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;
import Jama.*;

/*
 * @author Pierre-Louis bazin (pilou.bazin@uva.nl)
 *
 */
public class StackIntensityRegularisation {
	float[] image = null;
	
	int nx, ny, nz, nxyz;
	float rx, ry, rz;
	
	float cutoff = 50;
	
	float[] regularised;
	
	// set inputs
	public final void setInputImage(float[] val) { image = val; }
	
	public final void setVariationRatio(float val) { cutoff = val; }
	
	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }

	// outputs
	public final float[] getRegularisedImage() { return regularised; }

	public void execute() {
	    
	    // mask zero values
	    boolean[] mask = new boolean[nxyz];
	    for (int xyz=0;xyz<nxyz;xyz++) 
	        if (image[xyz]!=0) mask[xyz] = true;
            else mask[xyz] = false;
	    
	    // per slice:
	    double[] differences = new double [nx*ny];
	    int ndiff = 0;
	    for (int z=1;z<nz;z++) {
	        System.out.println("Processing slice "+z);
	        ndiff = 0;
	        for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) {
	            int xyz = x+nx*y+nx*ny*z;
	            int ngb = xyz-nx*ny;
	            if (mask[xyz] && mask[ngb]) {
	                differences[ndiff] = image[xyz]-image[ngb];
	                ndiff++;
	            }
	        }
	        if (ndiff>0) {
                // find the distribution excluding edges: only use the 50% central differences
                Percentile measure = new Percentile();
                double min = measure.evaluate(differences, 0, ndiff, 50-cutoff/2);
                double max = measure.evaluate(differences, 0, ndiff, 50+cutoff/2);
            
                // estimate the scaling factor (or curve)
                double[] curr = new double[ndiff];
                double[] prev = new double[ndiff];
                double mean = 0;
                int nkept=0;
                ndiff = 0;
                for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) {
                    int xyz = x+nx*y+nx*ny*z;
                    int ngb = xyz-nx*ny;
                    if (mask[xyz] && mask[ngb]) {
                        if (differences[ndiff]>=min && differences[ndiff]<=max) {
                            curr[nkept] = image[xyz];
                            prev[nkept] = image[ngb];
                            mean += image[xyz];
                            nkept++;
                        }
                        ndiff++;
                    }
                }
                if (nkept>0) {
                    mean /= (double)nkept;
                        
                    // linear least squares
                    double[][] fit = new double[nkept][1];
                    double[][] poly = new double[nkept][2];
                    for (int n=0;n<nkept;n++) {
                        fit[n][0] = curr[n];
                        poly[n][0] = 1.0;
                        poly[n][1] = prev[n];
                    }
                    // invert the linear model
                    Matrix mtx = new Matrix(poly);
                    Matrix smp = new Matrix(fit);
                    Matrix val = mtx.solve(smp);
                        
                    // compute the new values and residuals
                    double residual = 0;
                    double variance = 0;
                    nkept=0;
                    ndiff = 0;
                    for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) {
                        int xyz = x+nx*y+nx*ny*z;
                        int ngb = xyz-nx*ny;
                        if (mask[xyz]) {
                            // replace the image values directly -> possible drift? (shouldn't be the case)
                            double expected = val.get(0,0) + image[ngb]*val.get(1,0);
                            if (mask[ngb]) { 
                                if (differences[ndiff]>=min && differences[ndiff]<=max) {
                                    // compute residuals only where relevant
                                    variance += (image[xyz]-mean)*(image[xyz]-mean);
                                    residual += (image[xyz]-expected)*(image[xyz]-expected);
                                    nkept++;
                                }
                                ndiff++;
                            }
                            // change values
                            image[xyz] = (float)((image[xyz]-val.get(0,0))/val.get(1,0)); 
                        }
                    }
                    double rsquare = 1.0;
                    if (variance>0) rsquare = Numerics.max(1.0 - (residual/variance), 0.0);
                    System.out.println("bias: "+val.get(0,0));
                    System.out.println("scaling: "+val.get(1,0));
                    System.out.println("residuals R^2: "+rsquare);
                } else {
                    System.out.println("no good data: skip");
                }
            } else {
                System.out.println("empty mask overlap: skip");
            }
        }
	    // provide a global stabilisation? e.g. do the same process from the other direction?
	    // shouldn't be needed, hopefully..
	    regularised = image;
	    
		System.out.print("Done\n");
	}
}
