package nl.uva.imcn.algorithms;

import nl.uva.imcn.utilities.*;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.stat.descriptive.rank.Percentile;

/*
 * @author Pierre-Louis bazin (pilou.bazin@uva.nl)
 *
 */
public class T2sOptimalCombination {
	float[][] image = null;
	float[] te;
	int[] depth = null;
	
	int nx, ny, nz, nt, nxyz;
	float rx, ry, rz;
	
	int nimg = 4;
	float imax = 10000.0f;
	
	float[] combined;
	float[] s0img;
	float[] t2img;
	float[] r2img;
	float[] errimg;
	
	boolean robust = true;
	boolean local = false;
	
	// set inputs
	public final void setNumberOfEchoes(int val) { 
	    nimg = val;
	    image = new float[nimg][];
	    te = new float[nimg];
	}
	public final void setEchoImageAt(int num, float[] val) { image[num] = val; }
	public final void setEchoTimeAt(int num, float val) { te[num] = val; }
	
	public final void setImageEchoDepth(int[] val) { depth = val; }

	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nt=1; nxyz=nx*ny*nz; }
	public final void setDimensions(int x, int y, int z, int t) { nx=x; ny=y; nz=z; nt=t; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; if (dim.length>3) nt=dim[3]; else nt=1; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }

	// outputs
	public final float[] getCombinedImage() { return combined; }
	public final float[] getS0Image() { return s0img; }
	public final float[] getT2sImage() { return t2img; }
	public final float[] getR2sImage() { return r2img; }
	public final float[] getResidualImage() { return errimg; }

	public void execute() {
	    // fit the exponential
		double Sx, Sx2, Sy, Sxy, delta;
		Sx = 0.0f; Sx2 = 0.0f; delta = 0.0f;
		for (int i=0;i<nimg;i++) {
			Sx += -te[i];
			Sx2 += Numerics.square(-te[i]);
		}
		delta = nimg*Sx2 - Sx*Sx;
		
		System.out.print("Loading data\n");
		
		System.out.print("Echo times: [ ");
		for (int i=0;i<nimg;i++) System.out.print(te[i]+" ");
		System.out.print("]\n");
		
		combined = new float[nt*nxyz];
		s0img = new float[nxyz];
		r2img = new float[nxyz];
		t2img = new float[nxyz];
		errimg = new float[nxyz];
		for (int t=0;t<nt;t++) {
            double[] msqerr = new double[nimg];
            double den = 0.0;
		    for (int xyz=0;xyz<nxyz;xyz++) {
                boolean process=true;
                int dimg = nimg;
                if (depth!=null) dimg = Numerics.max(2, Numerics.min(nimg,depth[t]));
                 
                //for (int i=0;i<dimg;i++) if (image[i][xyz+nxyz*t]<=0) {
                    //process=false;
                //    image[i][xyz+nxyz*t] = 1;
                //}
                if (process) {
                    // average / median over entire time series?
                    // or process direction by direction?
                    s0img[xyz] = 0.0f;
                    r2img[xyz] = 0.0f;
                    t2img[xyz] = 0.0f;
                    errimg[xyz] = 0.0f;
                    
                    //System.out.print(".");
                    Sy = 0.0f;
                    Sxy = 0.0f;
                    for (int i=0;i<dimg;i++) {
                        double ydata = FastMath.log(1+image[i][xyz+nxyz*t]);
                        Sy += ydata;
                        Sxy += -ydata*te[i];
                    }
                    //s0img[xyz] = Numerics.bounded( (float)FastMath.exp( (Sx2*Sy-Sxy*Sx)/delta ), 0, imax);
                    double s0val = FastMath.exp( (Sx2*Sy-Sxy*Sx)/delta );
                    double r2val = Numerics.bounded( ( (nimg*Sxy-Sx*Sy)/delta ), 0.001, 1000.0);
                    
                    if (robust) {
                        // build pairwise estimates for Theil-Shen estimator
                        double[] slope = new double[dimg*(dimg-1)/2];
                        int p = 0;
                        for (int i=0;i<dimg;i++) for (int j=i+1;j<dimg;j++) {
                            double sx = -te[i]-te[j];
                            double sx2 = te[i]*te[i]+te[j]*te[j];
                            double sy = FastMath.log(1+image[i][xyz+nxyz*t]) + FastMath.log(1+image[j][xyz+nxyz*t]);
                            double sxy = -te[i]*FastMath.log(1+image[i][xyz+nxyz*t]) -te[j]*FastMath.log(1+image[j][xyz+nxyz*t]);
                            double dij = 2*sx2 - sx*sx;
                            double r2s =  (2*sxy-sx*sy)/dij;
                            slope[p] = r2s;
                            p++;
                        }
                        // find the median
                        Percentile measure = new Percentile();
                        r2val = Numerics.bounded( measure.evaluate(slope, 50.0), 0.001, 1000.0);
                        // get the corresponding SO
                        double[] intercept = new double[dimg];
                        for (int i=0;i<dimg;i++) {
                            intercept[i] = r2val*te[i] + FastMath.log(1+image[i][xyz+nxyz*t]);
                        }
                        s0val = FastMath.exp(measure.evaluate(intercept, 50.0));
                    }
                    double residual = 0.0;
                    double mean = 0.0;
                    double variance = 0.0;
                    for (int i=0;i<dimg;i++) mean += FastMath.log(1+image[i][xyz+nxyz*t]);
                    mean /= dimg;
                    for (int i=0;i<dimg;i++) {
                        double expected = FastMath.log(s0val) - te[i]*r2val;
                        variance += Numerics.square(mean-FastMath.log(1+image[i][xyz+nxyz*t]));
                        residual += Numerics.square(expected-FastMath.log(1+image[i][xyz+nxyz*t]));
                    }
                    double rsquare = 1.0;
                    if (variance>0) rsquare = Numerics.max(1.0 - (residual/variance), 0.0);
                    s0img[xyz] += (float)s0val;
                    r2img[xyz] += (float)r2val;
                    t2img[xyz] += 1.0f/(float)r2val;
                    errimg[xyz] = (float)rsquare;
                    
                    if (local) {
                        for (int i=0;i<nimg;i++) {
                            double expected = FastMath.log(s0val) - te[i]*r2val;
                            msqerr[i] += Numerics.square(expected-FastMath.log(1+image[i][xyz+nxyz*t]))/variance;
                        }
                    }
                    den++;
                }
			}
            if (local) {
                System.out.print("Echo quality:\n");
                for (int i=0;i<nimg;i++) {
                    System.out.print("Image "+t+" Echo "+i+" = "+FastMath.sqrt(msqerr[i]/den)+" ("+msqerr[i]+", "+den+")\n");
                }			
            }
		}
		if (!local) {
            // estimate echoes quality?
            double[][] msqerr = new double[nt][nimg];
            double den = 0.0;
            for (int xyz=0;xyz<nxyz;xyz++) {
                boolean process=true;
                //for (int i=0;i<nimg;i++) if (image[i][xyz]<=0) process=false;
                
                if (process) {
                    // now combine everything
                    s0img[xyz] /= nt;
                    r2img[xyz] /= nt;
                    t2img[xyz] /= nt;
                    errimg[xyz] /= nt;
                    
                    double[] weights = new double[nimg+1];
                    for (int i=0;i<nimg;i++) {
                        weights[i] = FastMath.exp(-r2img[xyz]*te[i]);
                        weights[nimg] += weights[i];
                    }
                    for (int t=0;t<nt;t++) {
                        for (int i=0;i<nimg;i++) {
                            msqerr[t][i] += Numerics.square(1+image[0][xyz+nxyz*t]-(1+image[i][xyz+nxyz*t])*weights[i]/weights[0]);
                        }
                    }
                    den++;
                }
            }
            System.out.print("Echo quality:\n");
            for (int t=0;t<nt;t++) for (int i=0;i<nimg;i++) {
                System.out.print("Image "+t+" Echo "+i+" = "+(msqerr[t][i]/den)+"\n");
            }
        }
        // global combination with variable depth
        for (int xyz=0;xyz<nxyz;xyz++) {
            boolean process=true;
            // now combine everything
            s0img[xyz] /= nt;
            r2img[xyz] /= nt;
            t2img[xyz] /= nt;
            errimg[xyz] /= nt;
            
            double[] weights = new double[nimg+1];
            for (int i=0;i<nimg;i++) {
                weights[i] = FastMath.exp(-r2img[xyz]*te[i]);
                weights[nimg] += weights[i];
            }
            
            for (int t=0;t<nt;t++) {
                int dimg = nimg;
                if (depth!=null) dimg = Numerics.max(2, Numerics.min(nimg,depth[t]));

                //for (int i=0;i<dimg;i++) if (image[i][xyz+nxyz*t]<=0) {
                //    process=false;
                //    image[i][xyz+nxyz*t]=1;
                //}
                if (process) {
                    combined[xyz+nxyz*t] = 0.0f;
                    double partialsum = 0.0;
                    double partialden = 0.0;
                    // kept depth: use image
                    for (int i=0;i<dimg;i++) {
                        combined[xyz+nxyz*t] += weights[i]/weights[nimg]*(1+image[i][xyz+nxyz*t]);
                        partialsum += (1+image[i][xyz+nxyz*t]);
                        partialden += weights[i];
                    }
                    // discarded depth: use combination of kept ones
                    for (int i=dimg;i<nimg;i++) {
                        combined[xyz+nxyz*t] += weights[i]/weights[nimg]*partialsum/partialden*weights[i];   
                    }
                }
            }
        }
		System.out.print("Done\n");
		
	}
}
