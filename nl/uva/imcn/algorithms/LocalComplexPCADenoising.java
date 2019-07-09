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
	private     int         nt = 1;

	private     int         nimg = 5;
	private		float		stdevCutoff = 1.1f;
	private     int         minDimension = 2;
	private     int         maxDimension = -1;
	private     int         ngbSize = 5;
	//private     int         winSize = -1;
	private     boolean     unwrap = true;
	private     boolean     eigen = false;
	//private     boolean     tvmag = false;
	//private     boolean     tvphs = false;
	private     boolean     slab2D = false;
	private     boolean     randomMatrix = false;
	
	//public static final String[]     thresholdingTypes = {"Eigenvalues","Global noise","Second half"};
	//private     String      thresholding = "Second half";
		
	// output parameters
	private float[][] images;
	private float[][] denoised;
	private float[][] eigvec;
	private float[][] eigval;
	private	float[] pcadim;
	private	float[] errmap;
	// TODO: add original noise level estimation (from the slope)
	
	// set inputs
	public final void setMagnitudeImageAt(int n, float[] in) {
	    if (invmag==null) invmag = new float[nimg*nt][nxyz];
	    for (int t=0;t<nt;t++) {
            for (int xyz=0;xyz<nxyz;xyz++) {
                invmag[t+n*nt][xyz] = in[xyz+t*nxyz];
            }
        }
	}
	
	public final void setPhaseImageAt(int n, float[] in) {
	    if (invphs==null) invphs = new float[nimg*nt][nxyz];
	    	    
	    for (int t=0;t<nt;t++) {
            for (int xyz=0;xyz<nxyz;xyz++) {
                invphs[t+n*nt][xyz] = in[xyz+t*nxyz];
            }
        }
    }
		
	public final void setImageNumber(int in) { nimg = in; }
	public final void setStdevCutoff(float in) { stdevCutoff = in; }
	public final void setMinimumDimension(int in) { minDimension = in; }
	public final void setMaximumDimension(int in) { maxDimension = in; }
	public final void setPatchSize(int in) { ngbSize = in; }
	//public final void setWindowSize(int in) { winSize = in; }
	public final void setUnwrapPhase(boolean in) { unwrap = in; }
	public final void setProcessSlabIn2D(boolean in) { slab2D = in; }
	public final void setRandomMatrixTheory(boolean in) { randomMatrix = in; }
	
	//public final void setMagnitudeTVSubtraction(boolean in) { tvmag = in; }
	//public final void setPhaseTVSubtraction(boolean in) { tvphs = in; }
	
	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; nt=1;}
	public final void setDimensions(int x, int y, int z, int t) { nx=x; ny=y; nz=z; nt=t; nxyz=nx*ny*nz;}
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; if (dim.length>3) nt=dim[3]; else nt=1; nxyz=nx*ny*nz;}
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }

	public final String getPackage() { return "IMCN Toolkit"; }
	public final String getCategory() { return "Intensity.devel"; } 
	public final String getLabel() { return "Local Complex PCA denoising"; }
	public final String getName() { return "Local Complex PCA denoising"; }

	public final String[] getAlgorithmAuthors() { return new String[]{"Pierre-Louis Bazin"}; }
	public final String getAffiliation() { return "Integrated Model-based Cognitive Neuroscience Reseaerch Unit, Universiteit van Amsterdam"; }
	public final String getDescription() { return "Denoise the data with a PCA-based method (adapted from Manjon et al., Plos One 2013."; }
		
	public final String getVersion() { return "1.0"; }

	// get outputs
	public float[] getDenoisedMagnitudeImageAt(int n) { 
	    float[] combi = new float[nxyz*nt];
	    for (int t=0;t<nt;t++) {
            for (int xyz=0;xyz<nxyz;xyz++) {
                combi[xyz+t*nxyz] = invmag[t+n*nt][xyz];
            }
	    }
	    return combi; 
	}
	public float[] getDenoisedPhaseImageAt(int n) {
	    float[] combi = new float[nxyz*nt];
	    for (int t=0;t<nt;t++) {
            for (int xyz=0;xyz<nxyz;xyz++) {
                combi[xyz+t*nxyz] = invphs[t+n*nt][xyz];
            }
	    }
	    return combi;
	}
	
	public float[] getLocalDimensionImage() { return pcadim; }
	public float[] getNoiseFitImage() { return errmap; }
	
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
		
		double[] phsscale = new double[nimg];
		for (int i=0;i<nimg;i++) phsscale[i] = 1.0;
		float[][] tvimgphs = null;
        if (unwrap) {
            // renormalize phase
            float[] phsmin = new float[nimg];
            float[] phsmax = new float[nimg];
            for (int i=0;i<nimg;i++) {
                phsmin[i] = invphs[i][0];
                phsmax[i] = invphs[i][0];
                for (int t=0;t<nt;t++) {
                    for (int xyz=0;xyz<nxyz;xyz++) {
                        if (invphs[t+i*nt][xyz]<phsmin[i]) phsmin[i] = invphs[t+i*nt][xyz];
                        if (invphs[t+i*nt][xyz]>phsmax[i]) phsmax[i] = invphs[t+i*nt][xyz];
                    }
                }
                phsscale[i] = (phsmax[i]-phsmin[i])/(2.0*FastMath.PI);
            }
            
            // unwrap phase and remove TV global variations
            //if (tvphs) {
            tvimgphs = new float[nimg*nt][];
            float[] phs = new float[nxyz];
            for (int i=0;i<nimg;i++) {
                for (int t=0;t<nt;t++) {
                    System.out.print("global variations removal phase "+(i+1)+", "+(t+1)+"\n");
                    for (int xyz=0;xyz<nxyz;xyz++) phs[xyz] = invphs[t+i*nt][xyz];
                     // unwrap phase images
                    FastMarchingPhaseUnwrapping unwrap = new FastMarchingPhaseUnwrapping();
                    unwrap.setPhaseImage(phs);
                    unwrap.setDimensions(nx,ny,nz);
                    unwrap.setResolutions(rx,ry,rz);
                    unwrap.setTVScale(0.33f);
                    unwrap.setTVPostProcessing("TV-approximation");
                    unwrap.execute();
                    tvimgphs[t+i*nt] = unwrap.getCorrectedImage();
                    for (int xyz=0;xyz<nxyz;xyz++) invphs[t+i*nt][xyz] = (float)(invphs[t+i*nt][xyz]/phsscale[i] - tvimgphs[t+i*nt][xyz]);
                }
            }
        } else {
            // still do the TV global variation removal
            tvimgphs = new float[nimg][];
            float[] phs = new float[nxyz];
            for (int i=0;i<nimg;i++) {
                for (int t=0;t<nt;t++) {
                    System.out.print("global variations removal phase "+(i+1)+", "+(t+1)+"\n");
                    for (int xyz=0;xyz<nxyz;xyz++) phs[xyz] = invphs[t+i*nt][xyz];
                    TotalVariation1D algo = new TotalVariation1D(phs,null,nx,ny,nz, 0.33f, 0.125f, 0.00001f, 500);
                    algo.setScaling((float)(2.0*FastMath.PI));
                    algo.solve();
                    tvimgphs[t+i*nt] = algo.exportResult();
                    for (int xyz=0;xyz<nxyz;xyz++) invphs[t+i*nt][xyz] -= tvimgphs[t+i*nt][xyz];
                }
            }
        }
		
		// 1. create all the sin, cos images
		images = new float[2*nimg*nt][nxyz];
		for (int i=0;i<nimg;i++) {
            for (int t=0;t<nt;t++) {
                for (int xyz=0;xyz<nxyz;xyz++) {
                    images[2*t+i*2*nt][xyz] = (float)(invmag[t+i*nt][xyz]*FastMath.cos(invphs[t+i*nt][xyz]));
                    images[2*t+1+i*2*nt][xyz] = (float)(invmag[t+i*nt][xyz]*FastMath.sin(invphs[t+i*nt][xyz]));
                }
            }
        }
        invmag = null;
        invphs = null;
		
		// 2. estimate PCA in slabs of NxNxN size xT windows
		int ngb = ngbSize;
		int nstep = Numerics.floor(ngb/2.0);
		int ndim = 2*nimg*nt;
		int hdim = nimg*nt;
		int nstepZ = nstep;
		if (slab2D) nstepZ = 1;
        System.out.print("patch dimensions ["+ngb+" x "+ndim+"] shifting by "+nstep+"\n");
		 
		denoised = new float[ndim][nxyz];
		if (eigen) {
            eigvec = new float[ndim][nxyz];
            eigval = new float[ndim][nxyz];
        } else {
            eigvec = null;
            eigval = null;
        }
		float[] weights = new float[nxyz];
		pcadim = new float[nxyz];
		errmap = new float[nxyz];
		// border issues should be cleaned-up, ignored so far
        for (int x=0;x<nx;x+=nstep) for (int y=0;y<ny;y+=nstep) for (int z=0;z<nz;z+=nstepZ) {
            int ngbx = Numerics.min(ngb, nx-x);
            int ngby = Numerics.min(ngb, ny-y);
            int ngbz = Numerics.min(ngb, nz-z);
            int ngb2 = ngbx*ngby;
            int ngb3 = ngbx*ngby*ngbz;
            boolean process = false;
            if (slab2D) {
                if (ngb2<ndim) {
                    //System.out.print("!patch is too small!\n");
                    process = false;
                } else {
                    process = true;
                }
            } else {
                if (ngb3<ndim) {
                    //System.out.print("!patch is too small!\n");
                    process = false;
                } else {
                    process = true;
                }
            }
            if (process) {
                double[][] patch;
                int ngbN;
                if (slab2D) {
                    ngbN=ngb2;
                    patch = new double[ngb2][ndim];
                    for (int dx=0;dx<ngbx;dx++) for (int dy=0;dy<ngby;dy++) for (int i=0;i<ndim;i++) {
                        patch[dx+ngbx*dy][i] = images[i][x+dx+nx*(y+dy)+nx*ny*z];
                    }
                } else {
                    ngbN = ngb3;
                    patch = new double[ngb3][ndim];
                    for (int dx=0;dx<ngbx;dx++) for (int dy=0;dy<ngby;dy++) for (int dz=0;dz<ngbz;dz++) for (int i=0;i<ndim;i++) {
                        patch[dx+ngbx*dy+ngbx*ngby*dz][i] = images[i][x+dx+nx*(y+dy)+nx*ny*(z+dz)];
                    }
                }
                // mean over samples
                double[] mean = new double[ndim];
                for (int i=0;i<ndim;i++) {
                   for (int n=0;n<ngbN;n++) mean[i] += patch[n][i];
                   mean[i] /= (double)ngbN;
                   for (int n=0;n<ngbN;n++) patch[n][i] -= mean[i];
                }
                // PCA from SVD X = USVt
                //System.out.println("perform SVD");
                Matrix M = new Matrix(patch);
                SingularValueDecomposition svd = M.svd();
            
                //System.out.println("eigenvalues: ");
                double[] eig = new double[ndim];
                boolean[] used = new boolean[ndim];
                int nzero=0;
                double eigsum = 0.0;
                for (int n=0;n<ndim;n++) {
                    eig[n] = svd.getSingularValues()[n];
                    eigsum += Numerics.abs(eig[n]);
                }
                double rsquare = 1.0;
                        
                // estimate noise
                if (randomMatrix) {
                    // use the random matrix theory algorithm of Veraart et al., 2016
                    
                    // 1. renormalize to eigenvalues
                    double[] eignorm = new double[ndim];
                    for (int n=0;n<ndim;n++) {
                        eignorm[n] = eig[n]*eig[n]/ngbN;
                    }
                    // 2. increase the number of used eigenvalues
                    int nkept = ndim-1;
         
                    for (int n=0;n<ndim-1;n++) {
                        double eignormsum = 0.0;
                        for (int m=n+1;m<ndim;m++) {
                            eignormsum += eignorm[m];
                        }
                        double sigma = (eignorm[n+1]-eignorm[ndim-1])/(4.0*FastMath.sqrt(ndim-n)/(double)ngbN);
                        if (eignormsum> (ndim-n)*sigma) {
                            nkept = n;
                            n = ndim;
                        }
                    }
                    for (int n=0;n<=nkept;n++) {
                        used[n] = true;   
                    }
                    for (int n=nkept+1;n<ndim;n++) {
                        used[n] = false;
                        nzero++;
                    }
                } else {
                    // simple version: compute the standard deviation of the patch
                    double sigma = 0.0;
                    for (int n=0;n<ngbN;n++) for (int i=0;i<ndim;i++) {
                        sigma += patch[n][i]*patch[n][i];
                    }
                    sigma /= ndim*ngbN;
                    sigma = FastMath.sqrt(sigma);
                    
                    // fit second half linear decay model
                    double[] loc = new double[hdim];
                    double[][] fit = new double[hdim][1];
                    for (int n=ndim-hdim;n<ndim;n++) {
                        loc[n-ndim+hdim] = (n-ndim+hdim)/(double)hdim;
                        fit[n-ndim+hdim][0] = Numerics.abs(eig[n]);
                    }
                    double[][] poly = new double[hdim][2];
                    for (int n=0;n<hdim;n++) {
                        poly[n][0] = 1.0;
                        poly[n][1] = loc[n];
                    }
                    // invert the linear model
                    Matrix mtx = new Matrix(poly);
                    Matrix smp = new Matrix(fit);
                    Matrix val = mtx.solve(smp);
            
                    // compute the expected value:
                    double[] expected = new double[ndim];
                    for (int n=0;n<ndim;n++) {
                        double n0 = (n-ndim+hdim)/(double)hdim;
                        // linear coeffs,
                        expected[n] = (val.get(0,0) + n0*val.get(1,0));
                        //expected[n] = n*slope + intercept;
                    }
                    double residual = 0.0;
                    double meaneig = 0.0;
                    double variance = 0.0;
                    for (int n=ndim-hdim;n<ndim;n++) meaneig += Numerics.abs(eig[n]);
                    meaneig /= hdim;
                    for (int n=ndim-hdim;n<ndim;n++) {
                        variance += (meaneig-Numerics.abs(eig[n]))*(meaneig-Numerics.abs(eig[n]));
                        residual += (expected[n]-Numerics.abs(eig[n]))*(expected[n]-Numerics.abs(eig[n]));
                    }
                    if (variance>0) rsquare = Numerics.max(1.0 - (residual/variance), 0.0);
                    
                    for (int n=0;n<ndim;n++) {
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
                }
                
                // reconstruct
                Matrix U = svd.getU();
                Matrix V = svd.getV();
                for (int n=0;n<ngbN;n++) for (int i=0;i<ndim;i++) {
                    // Sum_s>0 s_kU_kV_kt
                    patch[n][i] = mean[i];
                    for (int j=0;j<ndim;j++) if (used[j]) {
                        patch[n][i] += U.get(n,j)*eig[j]*V.get(i,j);
                    }
                }
                // add to the denoised image
                double wpatch = (1.0/(1.0 + ndim - nzero));
                for (int dx=0;dx<ngbx;dx++) for (int dy=0;dy<ngby;dy++) for (int dz=0;dz<ngbz;dz++) {
                    for (int i=0;i<ndim;i++) {
                        denoised[i][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*patch[dx+ngbx*dy+ngbx*ngby*dz][i]);
                        if (eigen) {
                            eigval[i][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*eig[i]);
                            eigvec[i][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*U.get(dx+ngbx*dy+ngbx*ngby*dz,i));
                        }
                    }
                    weights[x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)wpatch;
                    pcadim[x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*(ndim-nzero));
                    errmap[x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*rsquare);
                }
            }
        }
        for (int xyz=0;xyz<nxyz;xyz++) {
            for (int i=0;i<ndim;i++) {
                denoised[i][xyz] /= weights[xyz];
                if (eigen) {
                    eigval[i][xyz] /= weights[xyz];
                    eigvec[i][xyz] /= weights[xyz];
                }
            }
            pcadim[xyz] /= weights[xyz];
            errmap[xyz] /= weights[xyz];
        }
        //images = null;
          
        // 3. rebuild magnitude and phase images
        invmag = new float[nimg*nt][nxyz];
        invphs = new float[nimg*nt][nxyz];
  		for (int i=0;i<nimg;i++) {
  		    for (int t=0;t<nt;t++) {
                for (int xyz=0;xyz<nxyz;xyz++) {
                    invmag[t+i*nt][xyz] = (float)FastMath.sqrt(denoised[2*t+i*2*nt][xyz]*denoised[2*t+i*2*nt][xyz]+denoised[2*t+1+i*2*nt][xyz]*denoised[2*t+1+i*2*nt][xyz]);
                    //invphs[i][xyz] = (float)(FastMath.atan2(denoised[2*i+1][xyz],denoised[2*i][xyz])*phsscale[i]);
                    invphs[t+i*nt][xyz] = (float)FastMath.atan2(denoised[2*t+1+i*2*nt][xyz],denoised[2*t+i*2*nt][xyz]);
                 }
            }
        }
        
        // opt. add back the TV estimate
        for (int i=0;i<nimg;i++) for (int t=0;t<nt;t++) for (int xyz=0;xyz<nxyz;xyz++) {
            invphs[t+i*nt][xyz] += tvimgphs[t+i*nt][xyz];
            // wrap around phase values and rescale to original values
            invphs[t+i*nt][xyz] = (float)(Numerics.modulo(invphs[t+i*nt][xyz], 2.0*FastMath.PI)*phsscale[i]);
        }
	}

	private void executeMagnitudeDenoising() {
		// this assumes all the inputs are already set
		
		// main algorithm
		
		// we assume 4D images of size nimg
		if (invmag==null) System.out.print("data stacks not properly initialized!\n");
		
		// 1. pass directly the magnitude signal
		images = invmag;
        
		// 2. estimate PCA in slabs of NxNxN size xT windows
		int ngb = ngbSize;
		int nstep = Numerics.floor(ngb/2.0);
		int ndim = nimg*nt;
		int hdim = Numerics.ceil(ndim/2.0);
		int nstepZ = nstep;
		if (slab2D) nstepZ = 1;
		
		System.out.print("patch dimensions ["+ngb+" x "+ndim+"] shifting by "+nstep+"\n");
		 
		denoised = new float[ndim][nxyz];
		if (eigen) { 
		    eigvec = new float[ndim][nxyz];
		    eigval = new float[ndim][nxyz];
		} else {
		    eigvec = null;
		    eigval = null;
		}
		float[] weights = new float[nxyz];
		pcadim = new float[nxyz];
		errmap = new float[nxyz];
		// border issues should be cleaned-up, ignored so far
        for (int x=0;x<nx;x+=nstep) for (int y=0;y<ny;y+=nstep) for (int z=0;z<nz;z+=nstepZ) {
            int ngbx = Numerics.min(ngb, nx-x);
            int ngby = Numerics.min(ngb, ny-y);
            int ngbz = Numerics.min(ngb, nz-z);
            int ngb2 = ngbx*ngby;
            int ngb3 = ngbx*ngby*ngbz;
            boolean process = false;
            if (slab2D) {
                if (ngb2<ndim) {
                    //System.out.print("!patch is too small!\n");
                    process = false;
                } else {
                    process = true;
                }
            } else {
                if (ngb3<ndim) {
                    //System.out.print("!patch is too small!\n");
                    process = false;
                } else {
                    process = true;
                }
            }
            if (process) {
                double[][] patch;
                int ngbN;
                if (slab2D) {
                    ngbN=ngb2;
                    patch = new double[ngb2][ndim];
                    for (int dx=0;dx<ngbx;dx++) for (int dy=0;dy<ngby;dy++) for (int i=0;i<ndim;i++) {
                        patch[dx+ngbx*dy][i] = images[i][x+dx+nx*(y+dy)+nx*ny*z];
                    }
                } else {
                    ngbN = ngb3;
                    patch = new double[ngb3][ndim];
                    for (int dx=0;dx<ngbx;dx++) for (int dy=0;dy<ngby;dy++) for (int dz=0;dz<ngbz;dz++) for (int i=0;i<ndim;i++) {
                        patch[dx+ngbx*dy+ngbx*ngby*dz][i] = images[i][x+dx+nx*(y+dy)+nx*ny*(z+dz)];
                    }
                }
                // mean over samples
                double[] mean = new double[ndim];
                for (int i=0;i<nimg*nt;i++) {
                   for (int n=0;n<ngbN;n++) mean[i] += patch[n][i];
                   mean[i] /= (double)ngbN;
                   for (int n=0;n<ngbN;n++) patch[n][i] -= mean[i];
                }
                // PCA from SVD X = USVt
                //System.out.println("perform SVD");
                Matrix M = new Matrix(patch);
                SingularValueDecomposition svd = M.svd();
            
                // estimate noise
                // simple version: compute the standard deviation of the patch
                double sigma = 0.0;
                for (int n=0;n<ngbN;n++) for (int i=0;i<ndim;i++) {
                    sigma += patch[n][i]*patch[n][i];
                }
                sigma /= ndim*ngbN;
                sigma = FastMath.sqrt(sigma);
                
                // cutoff
                //System.out.println("eigenvalues: ");
                double[] eig = new double[ndim];
                boolean[] used = new boolean[ndim];
                int nzero=0;
                double eigsum = 0.0;
                for (int n=0;n<ndim;n++) {
                    eig[n] = svd.getSingularValues()[n];
                    eigsum += Numerics.abs(eig[n]);
                }
                // fit second half linear decay model
                double[] loc = new double[hdim];
                double[][] fit = new double[hdim][1];
                for (int n=ndim-hdim;n<ndim;n++) {
                    loc[n-ndim+hdim] = (n-ndim+hdim)/(double)hdim;
                    fit[n-ndim+hdim][0] = Numerics.abs(eig[n]);
                }
                double[][] poly = new double[hdim][2];
                for (int n=0;n<hdim;n++) {
                    poly[n][0] = 1.0;
                    poly[n][1] = loc[n];
                }
                // invert the linear model
                Matrix mtx = new Matrix(poly);
                Matrix smp = new Matrix(fit);
                Matrix val = mtx.solve(smp);
        
                // compute the expected value:
                double[] expected = new double[ndim];
                for (int n=0;n<ndim;n++) {
                    double n0 = (n-ndim+hdim)/(double)hdim;
                    // linear coeffs,
                    expected[n] = (val.get(0,0) + n0*val.get(1,0));
                    //expected[n] = n*slope + intercept;
                }
                double residual = 0.0;
                double meaneig = 0.0;
                double variance = 0.0;
                for (int n=ndim-hdim;n<ndim;n++) meaneig += Numerics.abs(eig[n]);
                meaneig /= hdim;
                for (int n=ndim-hdim;n<ndim;n++) {
                    variance += (meaneig-Numerics.abs(eig[n]))*(meaneig-Numerics.abs(eig[n]));
                    residual += (expected[n]-Numerics.abs(eig[n]))*(expected[n]-Numerics.abs(eig[n]));
                }
                double rsquare = 1.0;
                if (variance>0) rsquare = Numerics.max(1.0 - (residual/variance), 0.0);
                
                for (int n=0;n<ndim;n++) {
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
                for (int n=0;n<ngbN;n++) for (int i=0;i<ndim;i++) {
                    // Sum_s>0 s_kU_kV_kt
                    patch[n][i] = mean[i];
                    for (int j=0;j<ndim;j++) if (used[j]) {
                        patch[n][i] += U.get(n,j)*eig[j]*V.get(i,j);
                    }
                }
                // add to the denoised image
                double wpatch = (1.0/(1.0 + ndim - nzero));
                for (int dx=0;dx<ngbx;dx++) for (int dy=0;dy<ngby;dy++) for (int dz=0;dz<ngbz;dz++) {
                    for (int i=0;i<ndim;i++) {
                        denoised[i][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*patch[dx+ngbx*dy+ngbx*ngby*dz][i]);
                        if (eigen) {
                            eigval[i][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*eig[i]);
                            eigvec[i][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*U.get(dx+ngbx*dy+ngbx*ngby*dz,i));
                        }
                    }
                    weights[x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)wpatch;
                    pcadim[x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*(nimg-nzero));
                    errmap[x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*rsquare);
                }
            }
        }
        for (int xyz=0;xyz<nxyz;xyz++) {
            double err = 0.0;
            for (int i=0;i<ndim;i++) {
                denoised[i][xyz] /= weights[xyz];
                if (eigen) {
                    eigval[i][xyz] /= weights[xyz];
                    eigvec[i][xyz] /= weights[xyz];
                }
            }
            pcadim[xyz] /= weights[xyz];
            errmap[xyz] /= weights[xyz];
        }
        //images = null;
          
        // 3. rebuild magnitude and phase images
        invmag = denoised;

  		return;
	}

}
