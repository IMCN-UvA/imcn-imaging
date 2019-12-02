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
	private		float[][][] 	magnitude = null;
	private		float[][][] 	phase = null;
	private		int			nx, ny, nz, nt, nxyz, nmask;
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
	private	float[][] pcadim;
	private	float[][] errmap;
	
	private int[] index;
	private boolean[] mask;
	
	// TODO: add original noise level estimation (from the slope)
	
	// set inputs
	public final void setNumberOfContrasts(int val) { nc = val; }
	
	public final void setMaskImage(int[] val)  {
	    mask = new boolean[nxyz];
	    index = new int[nxyz];
	    nmask = 0;
	    for (int xyz=0;xyz<nxyz;xyz++) {
	        if (val[xyz]>0) {
	            mask[xyz] = true;
                index[xyz] = nmask;
                nmask++;
            } else {
                mask[xyz] = false;
                index[xyz] = -1;
            }
        }
	}
	    
	public final void setTimeSerieMagnitudeAt(int c, float[] in)  {
	    if (magnitude==null) {
	        magnitude = new float[nc][nmask][nt];
	    }
	    for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
	        for (int t=0;t<nt;t++) {
	            magnitude[c][index[xyz]][t] = in[xyz+t*nxyz];
	        }
	    }
	}
	public final void setTimeSeriePhaseAt(int c, float[] in)  {
	    if (phase==null) {
	        phase = new float[nc][nmask][nt];
	    }
	    for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
	        for (int t=0;t<nt;t++) {
	            phase[c][index[xyz]][t] = in[xyz+t*nxyz];
	        }
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
	public float[] getDenoisedMagnitudeAt(int c) {
	    float[] combi = new float[nxyz*nt];
	    for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
            for (int t=0;t<nt;t++) {
                combi[xyz+t*nxyz] = magnitude[c][index[xyz]][t];
            }
	    }
	    return combi;
	}
	
	public float[] getDenoisedPhaseAt(int c) {
	    float[] combi = new float[nxyz*nt];
	    for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
            for (int t=0;t<nt;t++) {
                combi[xyz+t*nxyz] = phase[c][index[xyz]][t];
            }
	    }
	    return combi;
	}
	
	public float[] getLocalDimensionImage() { return globalpcadim; }
	public float[] getNoiseFitImage() { return globalerrmap; }
	
	public void execute() {
	    if (phase==null) executeMagnitudeDenoising();
	    else executeComplexDenoising();
	}
	
	public void executeMagnitudeDenoising() {
		// this assumes all the inputs are already set
		
		// main algorithm
		
		// we assume nc 4D images of size nt
		if (magnitude==null) System.out.print("data stacks not properly initialized!\n");
		
		// 1. estimate PCA in slabs of NxNxN size CxT windows
		int ngb = ngbSize;
		int nstep = Numerics.floor(ngb/2.0);
		
		int ntime = winSize;
		int tstep = Numerics.floor(ntime/2.0);
		
		int nsample = Numerics.ceil(nt/tstep);
		
        System.out.print("patch dimensions ["+ngb+" x "+(ntime*nc)+"] shifting by ["+nstep+" x "+tstep+"]\n");
		System.out.print("time steps: "+nsample+" (over "+nt+" time points)\n");
		
		/* set beforehand
		// first get rid of all the masked values
		int nmask = 0;
		mask = new boolean[nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) {
		    mask[xyz] = false;
		    for (int t=0;t<nt;t++) for (int c=0;c<nc;c++) {
		        if (magnitude[c][xyz][t]!=0) {
		            mask[xyz] = true;
		            t = nt;
		            c = nc;
		        }
		    }
		    if (mask[xyz]) nmask++;
		}
		*/
		System.out.print("masking to "+(100.0*nmask/nxyz)+" percent of the image size\n");
		
		/*
		// build index file
		index = new int[nxyz];
		int id=0;
		for (int xyz=0;xyz<nxyz;xyz++) {
		    if (mask[xyz]) {
		        index[xyz] = id;
                id++;
            } else {
                index[xyz] = -1;
            }
		}*/
		
		/*
		// re-format the image array
		float[][][] maskedMagnitude = new float[nc][nmask][nt];
		id=0;
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int t=0;t<nt;t++) for (int c=0;c<nc;c++) {
		        maskedMagnitude[c][id][t] = magnitude[c][xyz][t];
		    }
		}
		magnitude = null;
		*/
		denoised = new float[nc][nmask][nt];
		//eigvec = new float[nimg][nxyz];
		//eigval = new float[nimg][nxyz];
		float[][] weights = new float[nmask][nt];
		pcadim = new float[nmask][nt];
		errmap = new float[nmask][nt];
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
                    int ngb3 = 0;
                    int[] patchid = new int[ngbx*ngby*ngbz];
                    for (int dx=0;dx<ngbx;dx++) for (int dy=0;dy<ngby;dy++) for (int dz=0;dz<ngbz;dz++) {
                        if (mask[x+dx+nx*(y+dy)+nx*ny*(z+dz)]) {
                            patchid[dx+ngbx*dy+ngbx*ngby*dz] = ngb3;
                            ngb3++;
                        } else {
                            patchid[dx+ngbx*dy+ngbx*ngby*dz] = -1;
                        }
                    }
                    
                    boolean process = false;
                    if (ngb3<ntime*nc) {
                        //System.out.print("!patch is too small!\n");
                        process = false;
                    } else {
                        process = true;
                    }
                    if (process) {
                        double[][] patch = new double[ngb3][ntime*nc];
                        for (int dx=0;dx<ngbx;dx++) for (int dy=0;dy<ngby;dy++) for (int dz=0;dz<ngbz;dz++) {
                            if (mask[x+dx+nx*(y+dy)+nx*ny*(z+dz)]) {
                                for (int ti=t;ti<t+ntime;ti++) for (int c=0;c<nc;c++) {
                                    //patch[dx+ngbx*dy+ngbx*ngby*dz][ti-t+c*ntime] = magnitude[c][x+dx+nx*(y+dy)+nx*ny*(z+dz)][ti];
                                    patch[patchid[dx+ngbx*dy+ngbx*ngby*dz]][ti-t+c*ntime] = magnitude[c][index[x+dx+nx*(y+dy)+nx*ny*(z+dz)]][ti];
                                }
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
                            if (mask[x+dx+nx*(y+dy)+nx*ny*(z+dz)]) {
                                for (int ti=0;ti<ntime;ti++) {
                                    for (int c=0;c<nc;c++) {
                                        denoised[c][index[x+dx+nx*(y+dy)+nx*ny*(z+dz)]][t+ti] += (float)(wpatch*patch[patchid[dx+ngbx*dy+ngbx*ngby*dz]][ti+c*ntime]);
                                        //denoised[c][x+dx+nx*(y+dy)+nx*ny*(z+dz)][t+ti] += (float)(wpatch*patch[dx+ngbx*dy+ngbx*ngby*dz][ti+c*ntime]);
                                        //eigval[t+i][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*eig[i]);
                                        //eigvec[t+i][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*U.get(dx+ngbx*dy+ngbx*ngby*dz,i));
                                        //weights[c][x+dx+nx*(y+dy)+nx*ny*(z+dz)][t+ti] += (float)wpatch;
                                        //pcadim[c][x+dx+nx*(y+dy)+nx*ny*(z+dz)][t+ti] += (float)(wpatch*(ntime-nzero));
                                        //errmap[c][x+dx+nx*(y+dy)+nx*ny*(z+dz)][t+ti] += (float)(wpatch*rsquare);
                                    }
                                    weights[index[x+dx+nx*(y+dy)+nx*ny*(z+dz)]][t+ti] += (float)wpatch;
                                    pcadim[index[x+dx+nx*(y+dy)+nx*ny*(z+dz)]][t+ti] += (float)(wpatch*(ntime-nzero));
                                    errmap[index[x+dx+nx*(y+dy)+nx*ny*(z+dz)]][t+ti] += (float)(wpatch*rsquare);
                                }
                                //weights[(t+i)/tstep][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)wpatch;
                                //pcadim[(t+i)/tstep][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*(ntime-nzero));
                                //errmap[(t+i)/tstep][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*rsquare);
                            }
                        }
                    }
                }
                if (last) t=nt;
            }
        }
        for (int ind=0;ind<nmask;ind++) {
            double err = 0.0;
            for (int t=0;t<nt;t++) {
                for (int c=0;c<nc;c++) {
                    denoised[c][ind][t] /= weights[ind][t];
                    //eigval[i][xyz] /= weights[i][xyz];
                    //eigvec[i][xyz] /= weights[i][xyz];
                    //pcadim[c][xyz][t] /= weights[c][xyz][t];
                    //errmap[c][xyz][t] /= weights[c][xyz][t];
                }
                pcadim[ind][t] /= weights[ind][t];
                errmap[ind][t] /= weights[ind][t];
            }
        }
        weights = null;
        magnitude = denoised;
        
       // magnitude = new float[nc][nxyz][nt];
        globalpcadim = new float[nt*nxyz];
        globalerrmap = new float[nxyz];
        for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
            globalerrmap[xyz] = 1.0f;
            for (int t=0;t<nt;t++) {
                //for (int c=0;c<nc;c++) {
                //    magnitude[c][xyz][t] = denoised[c][index[xyz]][t];
                //}
                globalpcadim[xyz+t*nxyz] = pcadim[index[xyz]][t];
                globalerrmap[xyz] = Numerics.min(globalerrmap[xyz], errmap[index[xyz]][t]);
            }
        }
  		return;
	}

	public void executeComplexDenoising() {
		// this assumes all the inputs are already set
		
		// main algorithm
		
		// we assume nc 4D images of size nt
		if (magnitude==null || phase==null) System.out.print("data stacks not properly initialized!\n");
		
		// Phase pre-processing
		
		// renormalize phase
        double phsscale = 1.0;
		float phsmin = 0.0f;
        float phsmax = 0.0f;
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int t=0;t<nt;t++) for (int c=0;c<nc;c++) {
		        if (phase[c][index[xyz]][t]<phsmin) phsmin = phase[c][index[xyz]][t];
		        if (phase[c][index[xyz]][t]>phsmax) phsmax = phase[c][index[xyz]][t];
		    }
		}
        phsscale = (phsmax-phsmin)/(2.0*FastMath.PI);
        
        // unwrap phase and remove TV global variations
        //if (tvphs) {
        float[] phs = new float[nxyz];
        float[] tv = new float[nxyz];
        float[][][] tvphs = new float[nc][nmask][nt];
        for (int c=0;c<nc;c++) {
            for (int t=0;t<nt;t++) {
                System.out.print("global variations removal phase "+(c+1)+", "+(t+1)+"\n");
                for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) phs[xyz] = phase[c][index[xyz]][t];
                 // unwrap phase images
                FastMarchingPhaseUnwrapping unwrap = new FastMarchingPhaseUnwrapping();
                unwrap.setPhaseImage(phs);
                unwrap.setMask(mask);
                unwrap.setDimensions(nx,ny,nz);
                unwrap.setResolutions(rx,ry,rz);
                unwrap.setTVScale(0.33f);
                unwrap.setTVPostProcessing("TV-approximation");
                unwrap.execute();
                tv = unwrap.getCorrectedImage();
                for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
                    phase[c][index[xyz]][t] = (float)(phs[xyz]/phsscale - tv[xyz]);
                    tvphs[c][index[xyz]][t] = tv[xyz];
                }
            }
        }
		
		// 1. create all the sin, cos images
		float[][][] images = new float[2*nc][nmask][nt];
		for (int c=0;c<nc;c++) {
            for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
                for (int t=0;t<nt;t++) {
                    images[2*c+0][index[xyz]][t] = (float)(magnitude[c][index[xyz]][t]*FastMath.cos(phase[c][index[xyz]][t]));
                    images[2*c+1][index[xyz]][t] = (float)(magnitude[c][index[xyz]][t]*FastMath.sin(phase[c][index[xyz]][t]));
                }
            }
        }
        magnitude = null;
        phase = null;		
		
		// 1. estimate PCA in slabs of NxNxN size CxT windows
		int ngb = ngbSize;
		int nstep = Numerics.floor(ngb/2.0);
		
		int ntime = winSize;
		int tstep = Numerics.floor(ntime/2.0);
		
		int nsample = Numerics.ceil(nt/tstep);
		
        System.out.print("patch dimensions ["+ngb+" x "+(ntime*2*nc)+"] shifting by ["+nstep+" x "+tstep+"]\n");
		System.out.print("time steps: "+nsample+" (over "+nt+" time points)\n");
		
		/* set beforehand
		// first get rid of all the masked values
		int nmask = 0;
		mask = new boolean[nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) {
		    mask[xyz] = false;
		    for (int t=0;t<nt;t++) for (int c=0;c<nc;c++) {
		        if (magnitude[c][xyz][t]!=0) {
		            mask[xyz] = true;
		            t = nt;
		            c = nc;
		        }
		    }
		    if (mask[xyz]) nmask++;
		}
		*/
		System.out.print("masking to "+(100.0*nmask/nxyz)+" percent of the image size\n");
		
		/*
		// build index file
		index = new int[nxyz];
		int id=0;
		for (int xyz=0;xyz<nxyz;xyz++) {
		    if (mask[xyz]) {
		        index[xyz] = id;
                id++;
            } else {
                index[xyz] = -1;
            }
		}*/
		
		/*
		// re-format the image array
		float[][][] maskedMagnitude = new float[nc][nmask][nt];
		id=0;
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int t=0;t<nt;t++) for (int c=0;c<nc;c++) {
		        maskedMagnitude[c][id][t] = magnitude[c][xyz][t];
		    }
		}
		magnitude = null;
		*/
		denoised = new float[2*nc][nmask][nt];
		//eigvec = new float[nimg][nxyz];
		//eigval = new float[nimg][nxyz];
		float[][] weights = new float[nmask][nt];
		pcadim = new float[nmask][nt];
		errmap = new float[nmask][nt];
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
                    int ngb3 = 0;
                    int[] patchid = new int[ngbx*ngby*ngbz];
                    for (int dx=0;dx<ngbx;dx++) for (int dy=0;dy<ngby;dy++) for (int dz=0;dz<ngbz;dz++) {
                        if (mask[x+dx+nx*(y+dy)+nx*ny*(z+dz)]) {
                            patchid[dx+ngbx*dy+ngbx*ngby*dz] = ngb3;
                            ngb3++;
                        } else {
                            patchid[dx+ngbx*dy+ngbx*ngby*dz] = -1;
                        }
                    }
                    
                    boolean process = false;
                    if (ngb3<ntime*2*nc) {
                        //System.out.print("!patch is too small!\n");
                        process = false;
                    } else {
                        process = true;
                    }
                    if (process) {
                        double[][] patch = new double[ngb3][ntime*2*nc];
                        for (int dx=0;dx<ngbx;dx++) for (int dy=0;dy<ngby;dy++) for (int dz=0;dz<ngbz;dz++) {
                            if (mask[x+dx+nx*(y+dy)+nx*ny*(z+dz)]) {
                                for (int ti=t;ti<t+ntime;ti++) for (int c=0;c<2*nc;c++) {
                                    //patch[dx+ngbx*dy+ngbx*ngby*dz][ti-t+c*ntime] = magnitude[c][x+dx+nx*(y+dy)+nx*ny*(z+dz)][ti];
                                    patch[patchid[dx+ngbx*dy+ngbx*ngby*dz]][ti-t+c*ntime] = images[c][index[x+dx+nx*(y+dy)+nx*ny*(z+dz)]][ti];
                                }
                            }
                        }
                        // mean over samples
                        double[] mean = new double[ntime*2*nc];
                        for (int i=0;i<ntime*2*nc;i++) {
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
                        for (int n=0;n<ngb3;n++) for (int i=0;i<ntime*2*nc;i++) {
                            sigma += patch[n][i]*patch[n][i];
                        }
                        sigma /= ntime*2*nc*ngb3;
                        sigma = FastMath.sqrt(sigma);
                        
                        // cutoff
                        //System.out.println("eigenvalues: ");
                        double[] eig = new double[ntime*2*nc];
                        boolean[] used = new boolean[ntime*2*nc];
                        int nzero=0;
                        double eigsum = 0.0;
                        for (int n=0;n<ntime*2*nc;n++) {
                            eig[n] = svd.getSingularValues()[n];
                            eigsum += Numerics.abs(eig[n]);
                        }
                        // fit second half linear decay model
                        int nfit = Numerics.floor(ntime*2*nc/2.0f);
                        double[] loc = new double[nfit];
                        double[][] fit = new double[nfit][1];
                        for (int n=ntime*2*nc-nfit;n<ntime*2*nc;n++) {
                            loc[n-ntime*2*nc+nfit] = (n-ntime*2*nc+nfit)/(double)nfit;
                            fit[n-ntime*2*nc+nfit][0] = Numerics.abs(eig[n]);
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
                        double[] expected = new double[ntime*2*nc];
                        for (int n=0;n<ntime*2*nc;n++) {
                            double n0 = (n-ntime*2*nc+nfit)/(double)nfit;
                            // linear coeffs,
                            expected[n] = (val.get(0,0) + n0*val.get(1,0));
                            //expected[n] = n*slope + intercept;
                        }
                        double residual = 0.0;
                        double meaneig = 0.0;
                        double variance = 0.0;
                        for (int n=nfit;n<ntime*2*nc;n++) meaneig += Numerics.abs(eig[n]);
                        meaneig /= (ntime*2*nc-nfit);
                        for (int n=nfit;n<ntime*nc;n++) {
                            variance += (meaneig-Numerics.abs(eig[n]))*(meaneig-Numerics.abs(eig[n]));
                            residual += (expected[n]-Numerics.abs(eig[n]))*(expected[n]-Numerics.abs(eig[n]));
                        }
                        double rsquare = 1.0;
                        if (variance>0) rsquare = Numerics.max(1.0 - (residual/variance), 0.0);
                        
                        for (int n=0;n<ntime*2*nc;n++) {
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
                        for (int n=0;n<ngb3;n++) for (int i=0;i<ntime*2*nc;i++) {
                            // Sum_s>0 s_kU_kV_kt
                            patch[n][i] = mean[i];
                            for (int j=0;j<ntime*2*nc;j++) if (used[j]) {
                                patch[n][i] += U.get(n,j)*eig[j]*V.get(i,j);
                            }
                        }
                        // add to the denoised image
                        double wpatch = (1.0/(1.0 + ntime*2*nc - nzero));
                        for (int dx=0;dx<ngbx;dx++) for (int dy=0;dy<ngby;dy++) for (int dz=0;dz<ngbz;dz++) {
                            if (mask[x+dx+nx*(y+dy)+nx*ny*(z+dz)]) {
                                for (int ti=0;ti<ntime;ti++) {
                                    for (int c=0;c<2*nc;c++) {
                                        denoised[c][index[x+dx+nx*(y+dy)+nx*ny*(z+dz)]][t+ti] += (float)(wpatch*patch[patchid[dx+ngbx*dy+ngbx*ngby*dz]][ti+c*ntime]);
                                        //denoised[c][x+dx+nx*(y+dy)+nx*ny*(z+dz)][t+ti] += (float)(wpatch*patch[dx+ngbx*dy+ngbx*ngby*dz][ti+c*ntime]);
                                        //eigval[t+i][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*eig[i]);
                                        //eigvec[t+i][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*U.get(dx+ngbx*dy+ngbx*ngby*dz,i));
                                        //weights[c][x+dx+nx*(y+dy)+nx*ny*(z+dz)][t+ti] += (float)wpatch;
                                        //pcadim[c][x+dx+nx*(y+dy)+nx*ny*(z+dz)][t+ti] += (float)(wpatch*(ntime-nzero));
                                        //errmap[c][x+dx+nx*(y+dy)+nx*ny*(z+dz)][t+ti] += (float)(wpatch*rsquare);
                                    }
                                    weights[index[x+dx+nx*(y+dy)+nx*ny*(z+dz)]][t+ti] += (float)wpatch;
                                    pcadim[index[x+dx+nx*(y+dy)+nx*ny*(z+dz)]][t+ti] += (float)(wpatch*(ntime*2*nc-nzero));
                                    errmap[index[x+dx+nx*(y+dy)+nx*ny*(z+dz)]][t+ti] += (float)(wpatch*rsquare);
                                }
                                //weights[(t+i)/tstep][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)wpatch;
                                //pcadim[(t+i)/tstep][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*(ntime-nzero));
                                //errmap[(t+i)/tstep][x+dx+nx*(y+dy)+nx*ny*(z+dz)] += (float)(wpatch*rsquare);
                            }
                        }
                    }
                }
                if (last) t=nt;
            }
        }
        images = null;
        
        for (int ind=0;ind<nmask;ind++) {
            double err = 0.0;
            for (int t=0;t<nt;t++) {
                for (int c=0;c<2*nc;c++) {
                    denoised[c][ind][t] /= weights[ind][t];
                    //eigval[i][xyz] /= weights[i][xyz];
                    //eigvec[i][xyz] /= weights[i][xyz];
                    //pcadim[c][xyz][t] /= weights[c][xyz][t];
                    //errmap[c][xyz][t] /= weights[c][xyz][t];
                }
                pcadim[ind][t] /= weights[ind][t];
                errmap[ind][t] /= weights[ind][t];
            }
        }
        weights = null;
        
        globalpcadim = new float[nt*nxyz];
        globalerrmap = new float[nxyz];
        for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
            globalerrmap[xyz] = 1.0f;
            for (int t=0;t<nt;t++) {
                globalpcadim[xyz+t*nxyz] = pcadim[index[xyz]][t];
                globalerrmap[xyz] = Numerics.min(globalerrmap[xyz], errmap[index[xyz]][t]);
            }
        }
        // rebuild magnitude and phase images
        magnitude = new float[nc][nmask][nt];
        phase = new float[nc][nmask][nt];
  		for (int c=0;c<nc;c++) {
  		    for (int t=0;t<nt;t++) {
                for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
                    magnitude[c][index[xyz]][t] = (float)FastMath.sqrt(denoised[2*c+0][index[xyz]][t]*denoised[2*c+0][index[xyz]][t]+denoised[2*c+1][index[xyz]][t]*denoised[2*c+1][index[xyz]][t]);
                    //invphs[i][xyz] = (float)(FastMath.atan2(denoised[2*i+1][xyz],denoised[2*i][xyz])*phsscale[i]);
                    phase[c][index[xyz]][t] = (float)FastMath.atan2(denoised[2*c+1][index[xyz]][t],denoised[2*c+0][index[xyz]][t]);
                 }
            }
        }
        
        // opt. add back the TV estimate
        for (int c=0;c<nc;c++) for (int t=0;t<nt;t++) for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
            phase[c][index[xyz]][t] += tvphs[c][index[xyz]][t];
            // wrap around phase values and rescale to original values
            phase[c][index[xyz]][t] = (float)(Numerics.modulo(phase[c][index[xyz]][t], 2.0*FastMath.PI)*phsscale);
        }
        
  		return;
	}
	
}
