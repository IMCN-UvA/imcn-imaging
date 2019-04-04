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
public class LocalComplexPCADenoising {

	// input parameters
	private		float[][] 	invmag = null;
	private		float[][] 	invphs = null;
	private		int			nx, ny, nz, nxyz;
	private 	float 		rx, ry, rz;

	private     int         nimg = 5;
	private		float		stdevCutoff = 1.1f;
	private     int         minDimension = 2;
	private     int         maxDimension = -1;
	private     int         ngbSize = 5;
	private     int         winSize = -1;
	private     boolean     unwrap = true;
	private     boolean     eigen = false;
	//private     boolean     tvmag = false;
	//private     boolean     tvphs = false;
	
	//public static final String[]     thresholdingTypes = {"Eigenvalues","Global noise","Second half"};
	//private     String      thresholding = "Second half";
		
	// output parameters
	private		float[] globalpcadim = null;
	private		float[] globalerrmap = null;
	
	// internal variables (for debug)
	private float[][] images;
	private float[][] denoised;
	private float[][] eigvec;
	private float[][] eigval;
	private	float[][] pcadim;
	private	float[][] errmap;
	// TODO: add original noise level estimation (from the slope)
	
	// set inputs
	public final void setMagnitudeImages(float[] in)  {
	    if (invmag==null) {
	        invmag = new float[nimg][nxyz];
	    }
	    for (int i=0;i<nimg;i++) {
            for (int xyz=0;xyz<nxyz;xyz++) {
                invmag[i][xyz] = in[xyz+i*nxyz];
            }
        }
	}
	public final void setPhaseImages(float[] in)  {
	    if (invphs==null) {
	        invphs = new float[nimg][nxyz];
	    }
	    for (int i=0;i<nimg;i++) {
            for (int xyz=0;xyz<nxyz;xyz++) {
                invphs[i][xyz] = in[xyz+i*nxyz+nimg*nxyz];
            }
        }
	}

	public final void setMagnitudeAndPhaseImage(float[] in) {
	    if (invmag==null || invphs==null) {
	        invmag = new float[nimg][nxyz];
	        invphs = new float[nimg][nxyz];
	    }
	    for (int i=0;i<nimg;i++) {
            for (int xyz=0;xyz<nxyz;xyz++) {
                invmag[i][xyz] = in[xyz+i*nxyz];
                invphs[i][xyz] = in[xyz+i*nxyz+nimg*nxyz];
            }
        }
	}
	
	public final void setMagnitudeImageAt(int n, float[] in) {
	    if (invmag==null) invmag = new float[nimg][];
	    
	    invmag[n] = in;
	}
	
	public final void setPhaseImageAt(int n, float[] in) {
	    if (invphs==null) invphs = new float[nimg][];
	    
	    invphs[n] = in;
	}
	
	public final void setImageNumber(int in) { nimg = in; }
	public final void setStdevCutoff(float in) { stdevCutoff = in; }
	public final void setMinimumDimension(int in) { minDimension = in; }
	public final void setMaximumDimension(int in) { maxDimension = in; }
	public final void setPatchSize(int in) { ngbSize = in; }
	public final void setWindowSize(int in) { winSize = in; }
	public final void setUnwrapPhase(boolean in) { unwrap = in; }
	
	//public final void setMagnitudeTVSubtraction(boolean in) { tvmag = in; }
	//public final void setPhaseTVSubtraction(boolean in) { tvphs = in; }
	
	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }

	public final String getPackage() { return "MICN Toolkit"; }
	public final String getCategory() { return "Intensity.devel"; } 
	public final String getLabel() { return "Local Complex PCA denoising"; }
	public final String getName() { return "Local Complex PCA denoising"; }

	public final String[] getAlgorithmAuthors() { return new String[]{"Pierre-Louis Bazin"}; }
	public final String getAffiliation() { return "Integrated Model-based Cognitive Neuroscience Reseaerch Unit, Universiteit van Amsterdam"; }
	public final String getDescription() { return "Denoise the data with a PCA-based method (adapted from Manjon et al., Plos One 2013."; }
		
	public final String getVersion() { return "1.0"; }

	// get outputs
	public float[] getDenoisedMagnitudeImageAt(int n) { return invmag[n]; }
	public float[] getDenoisedPhaseImageAt(int n) { return invphs[n]; }
	
	public float[] getDenoisedMagnitudeImages() {
	    float[] combi = new float[nxyz*nimg];
	    for (int i=0;i<nimg;i++) {
            for (int xyz=0;xyz<nxyz;xyz++) {
                combi[xyz+i*nxyz] = invmag[i][xyz];
            }
	    }
	    return combi;
	}
	public float[] getDenoisedPhaseImages(){
	    float[] combi = new float[nxyz*nimg];
	    for (int i=0;i<nimg;i++) {
            for (int xyz=0;xyz<nxyz;xyz++) {
                combi[xyz+i*nxyz] = invphs[i][xyz];
            }
	    }
	    return combi;
	}
	public float[] getDenoisedMagnitudeAndPhaseImage() {
	    float[] combi = new float[2*nxyz*nimg];
	    for (int i=0;i<nimg;i++) {
            for (int xyz=0;xyz<nxyz;xyz++) {
                combi[xyz+i*nxyz] = invmag[i][xyz];
                combi[xyz+i*nxyz+nimg*nxyz] = invphs[i][xyz];
            }
	    }
	    return combi;
	}
	
	public float[] getLocalDimensionImage() { return globalpcadim; }
	public float[] getNoiseFitImage() { return globalerrmap; }
	
	/*
	public float[] getRawComplexImage() {
	    float[] out = new float[2*nimg*nxyz];
	    for (int i=0;i<2*nimg;i++) for (int xyz=0;xyz<nxyz;xyz++) {
	        out[xyz+i*nxyz] = images[i][xyz];
	    }
	    return out;
	}

	public float[] getDenComplexImage() {
	    float[] out = new float[2*nimg*nxyz];
	    for (int i=0;i<2*nimg;i++) for (int xyz=0;xyz<nxyz;xyz++) {
	        out[xyz+i*nxyz] = denoised[i][xyz];
	    }
	    return out;
	}
	*/
	public float[] getEigenvectorImage() {
	    float[] out = new float[2*nimg*nxyz];
	    for (int i=0;i<2*nimg;i++) for (int xyz=0;xyz<nxyz;xyz++) {
	        out[xyz+i*nxyz] = eigvec[i][xyz];
	    }
	    return out;
	}

	public float[] getEigenvalueImage() {
	    float[] out = new float[2*nimg*nxyz];
	    for (int i=0;i<2*nimg;i++) for (int xyz=0;xyz<nxyz;xyz++) {
	        out[xyz+i*nxyz] = eigval[i][xyz];
	    }
	    return out;
	}
	
	public void execute() {
	    if (invphs==null) executeMagnitudeDenoising();
	    else executeComplexDenoising();
	}
	
	private void executeComplexDenoising() {
		// this assumes all the inputs are already set
		
		// main algorithm
		
		// we assume 4D images of size nimg
		if (invmag==null || invphs==null) System.out.print("data stacks not properly initialized!\n");
		
		double phsscale = 1.0;
		float[][] tvimgphs = null;
        if (unwrap) {
            // renormalize phase
            float phsmin = invphs[0][0];
            float phsmax = invphs[0][0];
            for (int i=0;i<nimg;i++) {
                for (int xyz=0;xyz<nxyz;xyz++) {
                    if (invphs[i][xyz]<phsmin) phsmin = invphs[i][xyz];
                    if (invphs[i][xyz]>phsmax) phsmax = invphs[i][xyz];
                }
            }
            phsscale = (phsmax-phsmin)/(2.0*FastMath.PI);
            
            // unwrap phase and remove TV global variations
            //if (tvphs) {
            tvimgphs = new float[nimg][];
            float[] phs = new float[nxyz];
            for (int i=0;i<nimg;i++) {
                System.out.print("global variations removal phase "+(i+1)+"\n");
                for (int xyz=0;xyz<nxyz;xyz++) phs[xyz] = invphs[i][xyz];
                 // unwrap phase images
                FastMarchingPhaseUnwrapping unwrap = new FastMarchingPhaseUnwrapping();
                unwrap.setPhaseImage(phs);
                unwrap.setDimensions(nx,ny,nz);
                unwrap.setResolutions(rx,ry,rz);
                unwrap.setTVScale(0.33f);
                unwrap.setTVPostProcessing("TV-approximation");
                unwrap.execute();
                tvimgphs[i] = unwrap.getCorrectedImage();
                for (int xyz=0;xyz<nxyz;xyz++) invphs[i][xyz] = (float)(invphs[i][xyz]/phsscale - tvimgphs[i][xyz]);
            }
        } else {
            // still do the TV global variation removal
            tvimgphs = new float[nimg][];
            float[] phs = new float[nxyz];
            for (int i=0;i<nimg;i++) {
                System.out.print("global variations removal phase "+(i+1)+"\n");
                for (int xyz=0;xyz<nxyz;xyz++) phs[xyz] = invphs[i][xyz];
                TotalVariation1D algo = new TotalVariation1D(phs,null,nx,ny,nz, 0.33f, 0.125f, 0.00001f, 500);
                algo.setScaling((float)(2.0*FastMath.PI));
                algo.solve();
                tvimgphs[i] = algo.exportResult();
                for (int xyz=0;xyz<nxyz;xyz++) invphs[i][xyz] -= tvimgphs[i][xyz];
            }
        }
		
		// 1. create all the sin, cos images
		images = new float[2*nimg][nxyz];
		for (int i=0;i<nimg;i++) {
            for (int xyz=0;xyz<nxyz;xyz++) {
                images[2*i][xyz] = (float)(invmag[i][xyz]*FastMath.cos(invphs[i][xyz]));
                images[2*i+1][xyz] = (float)(invmag[i][xyz]*FastMath.sin(invphs[i][xyz]));
            }
        }
        invmag = null;
        invphs = null;
		
		// 2. estimate PCA in slabs of NxNxN size xT windows
		int ngb = ngbSize;
		int nstep = Numerics.floor(ngb/2.0);
		int nimg2 = 2*nimg;
		boolean timeWindow=false;
		int ntime = Numerics.min(winSize,nimg);
		if (ntime<0) ntime = nimg;
		else timeWindow=true;
		int ntime2 = 2*ntime;
		int tstep = Numerics.floor(ntime/2.0);
		int nsample = Numerics.ceil(nimg/tstep);
        System.out.print("patch dimensions ["+ngb+" x "+ntime+"] shifting by ["+nstep+" x "+tstep+"]\n");
		if (timeWindow) System.out.print("time steps: "+nsample+" (over "+nimg+" time points)\n");
		 
		denoised = new float[nimg2][nxyz];
		if (eigen) {
            eigvec = new float[nimg2][nxyz];
            eigval = new float[nimg2][nxyz];
        } else {
            eigvec = null;
            eigval = null;
        }
		float[][] weights = new float[nimg][nxyz];
		pcadim = new float[nimg][nxyz];
		errmap = new float[nimg][nxyz];
		// border issues should be cleaned-up, ignored so far
        for (int t=0;t<nimg;t+=tstep) {
            boolean last = false;
		    boolean skip = false;
		    if (t+ntime>nimg) {
		        if (ntime<nimg) {
                    // shift windows at the end of the time domain if needed
                    t = nimg-ntime;
                    if (timeWindow) System.out.print("step "+(t/tstep)+":");
                    last = true;
                } else {
                    // skip this round entirely
                    //System.out.print("x\n");
                    skip = true;
                }
            } else {
                if (timeWindow) System.out.print("step "+(t/tstep));
            }
            if (!skip) {
                if (timeWindow) System.out.print("...\n");
                for (int x=0;x<nx;x+=nstep) for (int y=0;y<ny;y+=nstep) for (int z=0;z<nz;z+=nstep) {
                    int ngbx = Numerics.min(ngb, nx-x);
                    int ngby = Numerics.min(ngb, ny-y);
                    int ngbz = Numerics.min(ngb, nz-z);
                    int ngb3 = ngbx*ngby*ngbz;
                    boolean process = false;
                    if (ngb3<ntime2) {
                        System.out.print("!patch is too small!\n");
                        process = false;
                    } else {
                        process = true;
                    }
                    if (process) {
                        double[][] patch = new double[ngb3][ntime2];
                        for (int dx=0;dx<ngbx;dx++) for (int dy=0;dy<ngby;dy++) for (int dz=0;dz<ngbz;dz++) for (int i=2*t;i<2*t+ntime2;i++) {
                            patch[dx+ngbx*dy+ngbx*ngby*dz][i-2*t] = images[i][x+dx+nx*(y+dy)+nx*ny*(z+dz)];
                        }
                        // mean over samples
                        double[] mean = new double[ntime2];
                        for (int i=0;i<ntime2;i++) {
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
                        for (int n=0;n<ngb3;n++) for (int i=0;i<ntime2;i++) {
                            sigma += patch[n][i]*patch[n][i];
                        }
                        sigma /= ntime2*ngb3;
                        sigma = FastMath.sqrt(sigma);
                        
                        // cutoff
                        //System.out.println("eigenvalues: ");
                        double[] eig = new double[ntime2];
                        boolean[] used = new boolean[ntime2];
                        int nzero=0;
                        double eigsum = 0.0;
                        for (int n=0;n<ntime2;n++) {
                            eig[n] = svd.getSingularValues()[n];
                            eigsum += Numerics.abs(eig[n]);
                        }
                        // fit second half linear decay model
                        double[] loc = new double[ntime];
                        double[][] fit = new double[ntime][1];
                        for (int n=ntime;n<ntime2;n++) {
                            loc[n-ntime] = (n-ntime)/(double)ntime;
                            fit[n-ntime][0] = Numerics.abs(eig[n]);
                        }
                        double[][] poly = new double[ntime][2];
                        for (int n=0;n<ntime;n++) {
                            poly[n][0] = 1.0;
                            poly[n][1] = loc[n];
                        }
                        // invert the linear model
                        Matrix mtx = new Matrix(poly);
                        Matrix smp = new Matrix(fit);
                        Matrix val = mtx.solve(smp);
                
                        // compute the expected value:
                        double[] expected = new double[ntime2];
                        for (int n=0;n<ntime2;n++) {
                            double n0 = (n-ntime)/(double)ntime;
                            // linear coeffs,
                            expected[n] = (val.get(0,0) + n0*val.get(1,0));
                            //expected[n] = n*slope + intercept;
                        }
                        double residual = 0.0;
                        double meaneig = 0.0;
                        double variance = 0.0;
                        for (int n=ntime;n<ntime;n++) meaneig += Numerics.abs(eig[n]);
                        meaneig /= ntime;
                        for (int n=ntime;n<ntime2;n++) {
                            variance += (meaneig-Numerics.abs(eig[n]))*(meaneig-Numerics.abs(eig[n]));
                            residual += (expected[n]-Numerics.abs(eig[n]))*(expected[n]-Numerics.abs(eig[n]));
                        }
                        double rsquare = 1.0;
                        if (variance>0) rsquare = Numerics.max(1.0 - (residual/variance), 0.0);
                        
                        for (int n=0;n<ntime2;n++) {
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
                        for (int n=0;n<ngb3;n++) for (int i=0;i<ntime2;i++) {
                            // Sum_s>0 s_kU_kV_kt
                            patch[n][i] = mean[i];
                            for (int j=0;j<ntime2;j++) if (used[j]) {
                                patch[n][i] += U.get(n,j)*eig[j]*V.get(i,j);
                            }
                        }
                        // add to the denoised image
                        double wpatch = (1.0/(1.0 + ntime2 - nzero));
                        for (int dx=0;dx<ngbx;dx++) for (int dy=0;dy<ngby;dy++) for (int dz=0;dz<ngbz;dz++) {
                            for (int i=0;i<ntime2;i++) {
                                denoised[2*t+i][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*patch[dx+ngbx*dy+ngbx*ngby*dz][i]);
                                if (eigen) {
                                    eigval[2*t+i][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*eig[i]);
                                    eigvec[2*t+i][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*U.get(dx+ngbx*dy+ngbx*ngby*dz,i));
                                }
                            }
                            for (int i=0;i<ntime;i++) {
                                weights[t+i][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)wpatch;
                                pcadim[t+i][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*(ntime2-nzero));
                                errmap[t+i][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*rsquare);
                            }
                        }
                    }
                }
                if (last) t=nimg;
            }
        }
        for (int xyz=0;xyz<nxyz;xyz++) {
            double err = 0.0;
            for (int i=0;i<nimg2;i++) {
                int t = Numerics.floor(i/2.0f);
                denoised[i][xyz] /= weights[t][xyz];
                if (eigen) {
                    eigval[i][xyz] /= weights[t][xyz];
                    eigvec[i][xyz] /= weights[t][xyz];
                }
            }
            for (int i=0;i<nimg;i++) {
                pcadim[i][xyz] /= weights[i][xyz];
                errmap[i][xyz] /= weights[i][xyz];
            }
        }
        //images = null;
          
        // 3. rebuild magnitude and phase images
        invmag = new float[nimg][nxyz];
        invphs = new float[nimg][nxyz];
  		for (int i=0;i<nimg;i++) {
            for (int xyz=0;xyz<nxyz;xyz++) {
                invmag[i][xyz] = (float)FastMath.sqrt(denoised[2*i][xyz]*denoised[2*i][xyz]+denoised[2*i+1][xyz]*denoised[2*i+1][xyz]);
                invphs[i][xyz] = (float)(FastMath.atan2(denoised[2*i+1][xyz],denoised[2*i][xyz])*phsscale);
             }
        }
        
        if (ntime==nimg) {
            globalpcadim = new float[nxyz];
            globalerrmap = new float[nxyz];
            for (int xyz=0;xyz<nxyz;xyz++) {
                globalpcadim[xyz] = pcadim[0][xyz];
                globalerrmap[xyz] = errmap[0][xyz];
            }
        } else {
            globalpcadim = new float[nimg*nxyz];
            globalerrmap = new float[nimg*nxyz];
            for (int i=0;i<nimg;i++) {
                for (int xyz=0;xyz<nxyz;xyz++) {
                    globalpcadim[xyz+i*nxyz] = pcadim[i][xyz];
                    globalerrmap[xyz+i*nxyz] = errmap[i][xyz];
                }
            }
        }
        
        // opt. add back the TV estimate
        for (int i=0;i<nimg;i++) for (int xyz=0;xyz<nxyz;xyz++) {
            invphs[i][xyz] += tvimgphs[i][xyz];
            // wrap around phase values?
            invphs[i][xyz] = (float)(Numerics.modulo(invphs[i][xyz], 2.0*FastMath.PI)*phsscale);
        }
	}

	private void executeMagnitudeDenoising() {
		// this assumes all the inputs are already set
		
		// main algorithm
		
		// we assume 4D images of size nimg
		if (invmag==null) System.out.print("data stacks not properly initialized!\n");
		
		// 1. pass directly the magnitude signal
		/*
		images = new float[nimg][nxyz];
		for (int i=0;i<nimg;i++) {
            for (int xyz=0;xyz<nxyz;xyz++) {
                images[i][xyz] = invmag[i][xyz];
            }
        }
        invmag = null;
        */
        images = invmag;
        
		// 2. estimate PCA in slabs of NxNxN size xT windows
		int ngb = ngbSize;
		int nstep = Numerics.floor(ngb/2.0);
		
		boolean timeWindow = false;
		int ntime = Numerics.min(winSize,nimg);
		if (ntime<0) ntime = nimg;
		else timeWindow = true;
		int tstep = Numerics.floor(ntime/2.0);
		int nsample = Numerics.ceil(nimg/tstep);
        System.out.print("patch dimensions ["+ngb+" x "+ntime+"] shifting by ["+nstep+" x "+tstep+"]\n");
		if (timeWindow) System.out.print("time steps: "+nsample+" (over "+nimg+" time points)\n");
		 
		denoised = new float[nimg][nxyz];
		if (eigen) { 
		    eigvec = new float[nimg][nxyz];
		    eigval = new float[nimg][nxyz];
		} else {
		    eigvec = null;
		    eigval = null;
		}
		float[][] weights = new float[nimg][nxyz];
		pcadim = new float[nimg][nxyz];
		errmap = new float[nimg][nxyz];
		// border issues should be cleaned-up, ignored so far
		for (int t=0;t<nimg;t+=tstep) {
		    boolean last = false;
		    boolean skip = false;
		    if (t+ntime>nimg) {
		        if (ntime<nimg) {
                    // shift windows at the end of the time domain if needed
                    t = nimg-ntime;
                    if (timeWindow) System.out.print("step "+(t/tstep)+":");
                    last = true;
                } else {
                    // skip this round entirely
                    //System.out.print("x\n");
                    skip = true;
                }
            } else {
                if (timeWindow) System.out.print("step "+(t/tstep));
    		}
            if (!skip) {
                if (timeWindow) System.out.print("...\n");
                for (int x=0;x<nx;x+=nstep) for (int y=0;y<ny;y+=nstep) for (int z=0;z<nz;z+=nstep) {
                    int ngbx = Numerics.min(ngb, nx-x);
                    int ngby = Numerics.min(ngb, ny-y);
                    int ngbz = Numerics.min(ngb, nz-z);
                    int ngb3 = ngbx*ngby*ngbz;
                    boolean process = false;
                    if (ngb3<ntime) {
                        System.out.print("!patch is too small!\n");
                        process = false;
                    } else {
                        process = true;
                    }
                    if (process) {
                        double[][] patch = new double[ngb3][ntime];
                        for (int dx=0;dx<ngbx;dx++) for (int dy=0;dy<ngby;dy++) for (int dz=0;dz<ngbz;dz++) for (int i=t;i<t+ntime;i++) {
                            patch[dx+ngbx*dy+ngbx*ngby*dz][i-t] = images[i][x+dx+nx*(y+dy)+nx*ny*(z+dz)];
                        }
                        // mean over samples
                        double[] mean = new double[ntime];
                        for (int i=0;i<ntime;i++) {
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
                        for (int n=0;n<ngb3;n++) for (int i=0;i<ntime;i++) {
                            sigma += patch[n][i]*patch[n][i];
                        }
                        sigma /= ntime*ngb3;
                        sigma = FastMath.sqrt(sigma);
                        
                        // cutoff
                        //System.out.println("eigenvalues: ");
                        double[] eig = new double[ntime];
                        boolean[] used = new boolean[ntime];
                        int nzero=0;
                        double eigsum = 0.0;
                        for (int n=0;n<ntime;n++) {
                            eig[n] = svd.getSingularValues()[n];
                            eigsum += Numerics.abs(eig[n]);
                        }
                        // fit second half linear decay model
                        int ntimeh = Numerics.ceil(ntime/2.0f);
                        double[] loc = new double[ntimeh];
                        double[][] fit = new double[ntimeh][1];
                        for (int n=ntime-ntimeh;n<ntime;n++) {
                            loc[n-ntime+ntimeh] = (n-ntime+ntimeh)/(double)ntimeh;
                            fit[n-ntime+ntimeh][0] = Numerics.abs(eig[n]);
                        }
                        double[][] poly = new double[ntimeh][2];
                        for (int n=0;n<ntimeh;n++) {
                            poly[n][0] = 1.0;
                            poly[n][1] = loc[n];
                        }
                        // invert the linear model
                        Matrix mtx = new Matrix(poly);
                        Matrix smp = new Matrix(fit);
                        Matrix val = mtx.solve(smp);
                
                        // compute the expected value:
                        double[] expected = new double[ntime];
                        for (int n=0;n<ntime;n++) {
                            double n0 = (n-ntime+ntimeh)/(double)ntimeh;
                            // linear coeffs,
                            expected[n] = (val.get(0,0) + n0*val.get(1,0));
                            //expected[n] = n*slope + intercept;
                        }
                        double residual = 0.0;
                        double meaneig = 0.0;
                        double variance = 0.0;
                        for (int n=ntimeh;n<ntime;n++) meaneig += Numerics.abs(eig[n]);
                        meaneig /= (ntime-ntimeh);
                        for (int n=ntimeh;n<ntime;n++) {
                            variance += (meaneig-Numerics.abs(eig[n]))*(meaneig-Numerics.abs(eig[n]));
                            residual += (expected[n]-Numerics.abs(eig[n]))*(expected[n]-Numerics.abs(eig[n]));
                        }
                        double rsquare = 1.0;
                        if (variance>0) rsquare = Numerics.max(1.0 - (residual/variance), 0.0);
                        
                        for (int n=0;n<ntime;n++) {
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
                        for (int n=0;n<ngb3;n++) for (int i=0;i<ntime;i++) {
                            // Sum_s>0 s_kU_kV_kt
                            patch[n][i] = mean[i];
                            for (int j=0;j<ntime;j++) if (used[j]) {
                                patch[n][i] += U.get(n,j)*eig[j]*V.get(i,j);
                            }
                        }
                        // add to the denoised image
                        double wpatch = (1.0/(1.0 + ntime - nzero));
                        for (int dx=0;dx<ngbx;dx++) for (int dy=0;dy<ngby;dy++) for (int dz=0;dz<ngbz;dz++) {
                            for (int i=0;i<ntime;i++) {
                                denoised[t+i][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*patch[dx+ngbx*dy+ngbx*ngby*dz][i]);
                                if (eigen) {
                                    eigval[t+i][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*eig[i]);
                                    eigvec[t+i][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*U.get(dx+ngbx*dy+ngbx*ngby*dz,i));
                                }
                                weights[t+i][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)wpatch;
                                pcadim[t+i][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*(ntime-nzero));
                                errmap[t+i][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*rsquare);
                            }
                        }
                    }
                }
                if (last) t=nimg;
            }
        }
        for (int xyz=0;xyz<nxyz;xyz++) {
            double err = 0.0;
            for (int i=0;i<nimg;i++) {
                denoised[i][xyz] /= weights[i][xyz];
                if (eigen) {
                    eigval[i][xyz] /= weights[i][xyz];
                    eigvec[i][xyz] /= weights[i][xyz];
                }
                pcadim[i][xyz] /= weights[i][xyz];
                errmap[i][xyz] /= weights[i][xyz];
            }
        }
        //images = null;
          
        // 3. rebuild magnitude and phase images
        /*
        invmag = new float[nimg][nxyz];
        for (int i=0;i<nimg;i++) {
            for (int xyz=0;xyz<nxyz;xyz++) {
                invmag[i][xyz] = denoised[i][xyz];
            }
        }
        */
        invmag = denoised;
        
        if (ntime==nimg) {
            globalpcadim = new float[nxyz];
            globalerrmap = new float[nxyz];
            for (int xyz=0;xyz<nxyz;xyz++) {
                globalpcadim[xyz] = pcadim[0][xyz];
                globalerrmap[xyz] = errmap[0][xyz];
            }
        } else {
            globalpcadim = new float[nimg*nxyz];
            globalerrmap = new float[nimg*nxyz];
            for (int i=0;i<nimg;i++) {
                for (int xyz=0;xyz<nxyz;xyz++) {
                    globalpcadim[xyz+i*nxyz] = pcadim[i][xyz];
                    globalerrmap[xyz+i*nxyz] = errmap[i][xyz];
                }
            }
        }
  		return;
	}

}
