package nl.uva.imcn.algorithms;

import nl.uva.imcn.utilities.*;
import nl.uva.imcn.structures.*;
import nl.uva.imcn.libraries.*;
import nl.uva.imcn.methods.*;

import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.stat.descriptive.rank.*;

import Jama.*;

/*
 * @author Pierre-Louis Bazin
 */
public class LocalContrastAndTimeDenoising {

	// input parameters
	private		float[][][] 	images = null;
	private		int			nx, ny, nz, nt, nxyz;
	private 	float 		rx, ry, rz;

	private     int         nc = 3;
	private		float		stdevCutoff = 1.1f;
	private     int         minDimension = -1;
	private     int         maxDimension = -1;
	private     int         ngbSize = 3;
	private     int         winSize = 3;
		
	// output parameters
	private		float[] globalpcadim = null;
	private		float[] globalerrmap = null;
	
	// internal variables (for debug)
	private float[][][] denoised;
	private	float[][][] pcadim;
	private	float[][][] errmap;
	// TODO: add original noise level estimation (from the slope)
	
	// set inputs
	public final void setNumberOfContrasts(int val) { nc = val; }
	
	public final void setTimeSerieImageAt(int c, float[] in)  {
	    if (images==null) {
	        images = new float[nc][nxyz][nt];
	    }
	    for (int xyz=0;xyz<nxyz;xyz++) for (int t=0;t<nt;t++) {
            images[c][xyz][t] = in[xyz+t*nxyz];
        }
	}
	public final void setStdevCutoff(float in) { stdevCutoff = in; }
	public final void setMinimumDimension(int in) { minDimension = in; }
	public final void setMaximumDimension(int in) { maxDimension = in; }
	public final void setPatchSize(int in) { ngbSize = in; }
	public final void setWindowSize(int in) { winSize = in; }
	
	public final void setDimensions(int x, int y, int z, int t) { nx=x; ny=y; nz=z; nt=t; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nt=dim[3]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }

	public final String getPackage() { return "IMCN Toolkit"; }
	public final String getCategory() { return "Intensity.devel"; } 
	public final String getLabel() { return "Local Contrast and Time denoising"; }
	public final String getName() { return "Local Contrast and Time denoising"; }

	public final String[] getAlgorithmAuthors() { return new String[]{"Pierre-Louis Bazin"}; }
	public final String getAffiliation() { return "Integrated Model-based Cognitive Neuroscience Reseaerch Unit, Universiteit van Amsterdam"; }
	public final String getDescription() { return "Denoise multi-dimensional time series data with a PCA-based method (adapted from Manjon et al., Plos One 2013."; }
		
	public final String getVersion() { return "1.0"; }

	// get outputs
	public float[] getDenoisedImageAt(int c) {
	    float[] combi = new float[nxyz*nt];
	    for (int xyz=0;xyz<nxyz;xyz++) {
            for (int t=0;t<nt;t++) {
                combi[xyz+t*nxyz] = images[c][xyz][t];
            }
	    }
	    return combi;
	}
	
	public float[] getLocalDimensionImage() { return globalpcadim; }
	public float[] getNoiseFitImage() { return globalerrmap; }
	
	
	public void execute() {
		// this assumes all the inputs are already set
		
		// main algorithm
		
		// we assume nc 4D images of size nt
		if (images==null) System.out.print("data stacks not properly initialized!\n");
		
		// 1. estimate PCA in slabs of NxNxN size CxT windows
		int ngb = ngbSize;
		int nstep = Numerics.floor(ngb/2.0);
		
		int ntime = winSize;
		int tstep = Numerics.floor(ntime/2.0);
		
		int nsample = Numerics.ceil(nt/tstep);
		
        System.out.print("patch dimensions ["+ngb+" x "+(ntime*nc)+"] shifting by ["+nstep+" x "+tstep+"]\n");
		System.out.print("time steps: "+nsample+" (over "+nt+" time points)\n");
		 
		denoised = new float[nc][nxyz][nt];
		//eigvec = new float[nimg][nxyz];
		//eigval = new float[nimg][nxyz];
		float[][][] weights = new float[nc][nxyz][nt];
		pcadim = new float[nc][nxyz][nt];
		errmap = new float[nc][nxyz][nt];
		// border issues should be cleaned-up, ignored so far
		for (int t=0;t<nt;t+=tstep) {
		    boolean last = false;
		    boolean skip = false;
		    if (t+ntime>nt) {
		        if (ntime<nt) {
                    // shift windows at the end of the time domain if needed
                    t = nt-ntime;
                    System.out.print("step "+(t/tstep)+":");
                    last = true;
                } else {
                    // skip this round entirely
                    //System.out.print("x\n");
                    skip = true;
                }
            } else {
                System.out.print("step "+(t/tstep));
    		}
            if (!skip) {
                System.out.print("...\n");
                for (int x=0;x<nx;x+=nstep) for (int y=0;y<ny;y+=nstep) for (int z=0;z<nz;z+=nstep) {
                    int ngbx = Numerics.min(ngb, nx-x);
                    int ngby = Numerics.min(ngb, ny-y);
                    int ngbz = Numerics.min(ngb, nz-z);
                    int ngb3 = ngbx*ngby*ngbz;
                    boolean process = false;
                    if (ngb3<ntime*nc) {
                        System.out.print("!patch is too small!\n");
                        process = false;
                    } else {
                        process = true;
                    }
                    if (process) {
                        double[][] patch = new double[ngb3][ntime*nc];
                        for (int dx=0;dx<ngbx;dx++) for (int dy=0;dy<ngby;dy++) for (int dz=0;dz<ngbz;dz++) {
                            for (int ti=t;ti<t+ntime;ti++) for (int c=0;c<nc;c++) {
                                patch[dx+ngbx*dy+ngbx*ngby*dz][ti-t+c*ntime] = images[c][x+dx+nx*(y+dy)+nx*ny*(z+dz)][ti];
                            }
                        }
                        // mean over samples
                        double[] mean = new double[ntime*nc];
                        for (int i=0;i<ntime*nc;i++) {
                           for (int n=0;n<ngb3;n++) mean[i] += patch[n][i];
                           mean[i] /= (double)ngb3;
                           for (int n=0;n<ngb3;n++) patch[n][i] -= mean[i];
                        }
                        // PCA from SVD X = USVt
                        //System.out.println("perform SVD");
                        Matrix M = new Matrix(patch);
                        SingularValueDecomposition svd = M.svd();
                    
                        // estimate noise
                        // simple version: compute the standard deviation of the patch
                        double sigma = 0.0;
                        for (int n=0;n<ngb3;n++) for (int i=0;i<ntime*nc;i++) {
                            sigma += patch[n][i]*patch[n][i];
                        }
                        sigma /= ntime*nc*ngb3;
                        sigma = FastMath.sqrt(sigma);
                        
                        // cutoff
                        //System.out.println("eigenvalues: ");
                        double[] eig = new double[ntime*nc];
                        boolean[] used = new boolean[ntime*nc];
                        int nzero=0;
                        double eigsum = 0.0;
                        for (int n=0;n<ntime*nc;n++) {
                            eig[n] = svd.getSingularValues()[n];
                            eigsum += Numerics.abs(eig[n]);
                        }
                        // fit second half linear decay model
                        int nfit = Numerics.floor(ntime*nc/2.0f);
                        double[] loc = new double[nfit];
                        double[][] fit = new double[nfit][1];
                        for (int n=ntime*nc-nfit;n<ntime*nc;n++) {
                            loc[n-ntime*nc+nfit] = (n-ntime*nc+nfit)/(double)nfit;
                            fit[n-ntime*nc+nfit][0] = Numerics.abs(eig[n]);
                        }
                        double[][] poly = new double[nfit][2];
                        for (int n=0;n<nfit;n++) {
                            poly[n][0] = 1.0;
                            poly[n][1] = loc[n];
                        }
                        // invert the linear model
                        Matrix mtx = new Matrix(poly);
                        Matrix smp = new Matrix(fit);
                        Matrix val = mtx.solve(smp);
                
                        // compute the expected value:
                        double[] expected = new double[ntime*nc];
                        for (int n=0;n<ntime*nc;n++) {
                            double n0 = (n-ntime*nc+nfit)/(double)nfit;
                            // linear coeffs,
                            expected[n] = (val.get(0,0) + n0*val.get(1,0));
                            //expected[n] = n*slope + intercept;
                        }
                        double residual = 0.0;
                        double meaneig = 0.0;
                        double variance = 0.0;
                        for (int n=nfit;n<ntime*nc;n++) meaneig += Numerics.abs(eig[n]);
                        meaneig /= (ntime*nc-nfit);
                        for (int n=nfit;n<ntime*nc;n++) {
                            variance += (meaneig-Numerics.abs(eig[n]))*(meaneig-Numerics.abs(eig[n]));
                            residual += (expected[n]-Numerics.abs(eig[n]))*(expected[n]-Numerics.abs(eig[n]));
                        }
                        double rsquare = 1.0;
                        if (variance>0) rsquare = Numerics.max(1.0 - (residual/variance), 0.0);
                        
                        for (int n=0;n<ntime*nc;n++) {
                            //System.out.print(" "+(eig[n]/sigma));
                            if (n>=minDimension && Numerics.abs(eig[n]) < stdevCutoff*expected[n]) {
                                used[n] = false;
                                nzero++;
                                //System.out.print("(-),");
                            } else  if (maxDimension>0 && n>=maxDimension) {
                                used[n] = false;
                                nzero++;
                                //System.out.print("(|),");
                            } else {
                                used[n] = true;
                                //System.out.print("(+),");
                            }
                        }
                        // reconstruct
                        Matrix U = svd.getU();
                        Matrix V = svd.getV();
                        for (int n=0;n<ngb3;n++) for (int i=0;i<ntime*nc;i++) {
                            // Sum_s>0 s_kU_kV_kt
                            patch[n][i] = mean[i];
                            for (int j=0;j<ntime*nc;j++) if (used[j]) {
                                patch[n][i] += U.get(n,j)*eig[j]*V.get(i,j);
                            }
                        }
                        // add to the denoised image
                        double wpatch = (1.0/(1.0 + ntime*nc - nzero));
                        for (int dx=0;dx<ngbx;dx++) for (int dy=0;dy<ngby;dy++) for (int dz=0;dz<ngbz;dz++) {
                            for (int ti=0;ti<ntime;ti++) for (int c=0;c<nc;c++) {
                                denoised[c][x+dx+nx*(y+dy)+nx*ny*(z+dz)][t+ti] += (float)(wpatch*patch[dx+ngbx*dy+ngbx*ngby*dz][ti+c*ntime]);
                                //eigval[t+i][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*eig[i]);
                                //eigvec[t+i][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*U.get(dx+ngbx*dy+ngbx*ngby*dz,i));
                                weights[c][x+dx+nx*(y+dy)+nx*ny*(z+dz)][t+ti] += (float)wpatch;
                                pcadim[c][x+dx+nx*(y+dy)+nx*ny*(z+dz)][t+ti] += (float)(wpatch*(ntime-nzero));
                                errmap[c][x+dx+nx*(y+dy)+nx*ny*(z+dz)][t+ti] += (float)(wpatch*rsquare);
                            }
                            //weights[(t+i)/tstep][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)wpatch;
                            //pcadim[(t+i)/tstep][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*(ntime-nzero));
                            //errmap[(t+i)/tstep][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*rsquare);
                        }
                    }
                }
                if (last) t=nt;
            }
        }
        for (int xyz=0;xyz<nxyz;xyz++) {
            double err = 0.0;
            for (int t=0;t<nt;t++) for (int c=0;c<nc;c++) {
                denoised[c][xyz][t] /= weights[c][xyz][t];
                //eigval[i][xyz] /= weights[i][xyz];
                //eigvec[i][xyz] /= weights[i][xyz];
                pcadim[c][xyz][t] /= weights[c][xyz][t];
                errmap[c][xyz][t] /= weights[c][xyz][t];
            }
        }
        images = denoised;
        
        globalpcadim = new float[nt*nxyz];
        globalerrmap = new float[nt*nxyz];
        for (int t=0;t<nt;t++) {
            for (int xyz=0;xyz<nxyz;xyz++) {
                globalpcadim[xyz+t*nxyz] = pcadim[0][xyz][t];
                globalerrmap[xyz+t*nxyz] = errmap[0][xyz][t];
            }
        }
  		return;
	}

}
