package nl.uva.imcn.algorithms;

import nl.uva.imcn.utilities.*;
import org.apache.commons.math3.util.FastMath;

/*
 * @author Pierre-Louis bazin (pilou.bazin@uva.nl)
 *
 */
public class T2sOptimalCombination {
	float[][] image = null;
	float[] te;
	
	int nx, ny, nz, nt, nxyz;
	float rx, ry, rz;
	
	int nimg = 4;
	float imax = 10000.0f;
	
	float[] combined;
	float[] s0img;
	float[] t2img;
	float[] r2img;
	float[] errimg;
	
	// set inputs
	public final void setNumberOfEchoes(int val) { 
	    nimg = val;
	    image = new float[nimg][];
	    te = new float[nimg];
	}
	public final void setEchoImageAt(int num, float[] val) { image[num] = val; }
	public final void setEchoTimeAt(int num, float val) { te[num] = val; }

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
		
		s0img = new float[nxyz];
		r2img = new float[nxyz];
		t2img = new float[nxyz];
		errimg = new float[nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) {
			boolean process=true;
			for (int i=0;i<nimg;i++) if (image[i][xyz]==0) process=false;
			if (process) {
			    // average over entire time series
			    s0img[xyz] = 0.0f;
			    r2img[xyz] = 0.0f;
			    t2img[xyz] = 0.0f;
			    errimg[xyz] = 0.0f;
			    for (int t=0;t<nt;t++) {
                    //System.out.print(".");
                    Sy = 0.0f;
                    Sxy = 0.0f;
                    for (int i=0;i<nimg;i++) {
                        double ydata = FastMath.log(image[i][xyz+nxyz*t]);
                        Sy += ydata;
                        Sxy += -ydata*te[i];
                    }
                    //s0img[xyz] = Numerics.bounded( (float)FastMath.exp( (Sx2*Sy-Sxy*Sx)/delta ), 0, imax);
                    double s0val = FastMath.exp( (Sx2*Sy-Sxy*Sx)/delta );
                    double r2val = Numerics.bounded( ( (nimg*Sxy-Sx*Sy)/delta ), 0.001, 1.0);
                    double residual = 0.0;
                    double mean = 0.0;
                    double variance = 0.0;
                    for (int i=0;i<nimg;i++) mean += FastMath.log(image[i][xyz+nxyz*t]);
                    mean /= nimg;
                    for (int i=0;i<nimg;i++) {
                        double expected = FastMath.log(s0val) - te[i]*r2val;
                        variance += Numerics.square(mean-FastMath.log(image[i][xyz+nxyz*t]));
                        residual += Numerics.square(expected-FastMath.log(image[i][xyz+nxyz*t]));
                    }
                    double rsquare = 1.0;
                    if (variance>0) rsquare = Numerics.max(1.0 - (residual/variance), 0.0);
                    s0img[xyz] += (float)s0val;
                    r2img[xyz] += (float)r2val;
                    t2img[xyz] += 1.0f/(float)r2val;
                    errimg[xyz] = (float)rsquare;        
                }
			}
		}
		for (int xyz=0;xyz<nxyz;xyz++) {
			boolean process=true;
			for (int i=0;i<nimg;i++) if (image[i][xyz]==0) process=false;
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
			        combined[xyz+nxyz*t] = 0.0f;
			        for (int i=0;i<nimg;i++) {
			            combined[xyz+nxyz*t] += weights[i]/weights[nimg]*image[i][xyz+nxyz*t];
			        }
			    }
			}
		}
		System.out.print("Done\n");
		
	}
}
