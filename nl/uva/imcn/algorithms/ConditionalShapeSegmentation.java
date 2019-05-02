package nl.uva.imcn.algorithms;

import nl.uva.imcn.utilities.*;
import nl.uva.imcn.structures.*;
import nl.uva.imcn.libraries.*;
import nl.uva.imcn.methods.*;

import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.stat.descriptive.rank.*;

//import Jama.*;

/*
 * @author Pierre-Louis Bazin
 */
public class ConditionalShapeSegmentation {

	// data buffers
	private float[][][] lvlImages;
	private float[][][] intensImages;
	
	private float[][] targetImages;
	
	private double[][][] condmean = null;
	private double[][][] condstdv = null;
	
	private boolean[] mask = null;;
	
	private int nsub;
	private int nobj;
	private int nc;
	private int nbest = 16;
	
	private float deltaIn = 2.0f;
	private float deltaOut = 0.0f;
	private float boundary = 5.0f;
	private boolean modelBackground = true;
	private boolean cancelBackground = true;
	private boolean cancelAll = false;
	private boolean sumPosterior = false;
	private boolean maxPosterior = false;
	private int maxiter = 0;
	private float maxdiff = 0.01f;
	//private boolean topoParam = true;
	//private     String	            lutdir = null;
	private double top = 95.0;
	private boolean rescaleProbas = true;
	private boolean rescaleIntensities = true;
	private boolean modelHistogram = true;
	private boolean rescaleHistograms = true;
	
	
	// possibly to extend to entire distribution?
	// model size: nbins x nc x nobj x nobj
	// benefits: no probability computation, no a priori model
	private double[][][][] condhistogram = null;
	private double[][][] condmin = null;
	private double[][][] condmax = null;
	private int nbins=200;
	
	private boolean[][][] condpair = null;
	
	private float[] medstdv = null;
	
	private int[][]        spatialLabels;
	private float[][]      spatialProbas;
	
	private int[][]        intensityLabels;
	private float[][]      intensityProbas;
	
	private int[][]        combinedLabels;
	private float[][]      combinedProbas;
	
	private int[][]        diffusedLabels;
	private float[][]      diffusedProbas;
	
	private int[]          finalLabel;
	private float[]        finalProba;
	
	private float[][]      ngbw;
	private int[]          idmap;
	
	private float[]        objVolumeMean;
	private float[]        objVolumeStdv;
	
	private int nx, ny, nz, nxyz;
	private float rx, ry, rz;
	private int ndata;

	// in case the atlas and target spaces are different
	private float[]    map2atlas = null;
	private float[]    map2target = null;
	
	private int ntx, nty, ntz, ntxyz;
	private float rtx, rty, rtz;

	public final void setNumberOfSubjectsObjectsAndContrasts(int sub,int obj,int cnt) {
	    nsub = sub;
	    nobj = obj;
	    nc = cnt;
	    lvlImages = new float[nsub][nobj][];
	    intensImages = new float[nsub][nc][];
	    targetImages = new float[nc][];
	}
	public final void setLevelsetImageAt(int sub, int obj, float[] val) { lvlImages[sub][obj] = val; 
	    //System.out.println("levelset ("+sub+", "+obj+") = "+lvlImages[sub][obj][Numerics.floor(nx/2+nx*ny/2+nx*ny*nz/2)]);
	}
	public final void setContrastImageAt(int sub, int cnt, float[] val) { intensImages[sub][cnt] = val; 
	    //System.out.println("contrast ("+sub+", "+cnt+") = "+intensImages[sub][cnt][Numerics.floor(nx/2+nx*ny/2+nx*ny*nz/2)]);
	}
	public final void setTargetImageAt(int cnt, float[] val) { targetImages[cnt] = val; 
	    //System.out.println("target ("+cnt+") = "+targetImages[cnt][Numerics.floor(nx/2+nx*ny/2+nx*ny*nz/2)]);
	}
	public final void setMappingToTarget(float[] val) { map2target = val; 
	    //System.out.println("target ("+cnt+") = "+targetImages[cnt][Numerics.floor(nx/2+nx*ny/2+nx*ny*nz/2)]);
	}
	public final void setMappingToAtlas(float[] val) { map2atlas = val; 
	    //System.out.println("target ("+cnt+") = "+targetImages[cnt][Numerics.floor(nx/2+nx*ny/2+nx*ny*nz/2)]);
	}
	public final void setShapeAtlasProbasAndLabels(float[] pval, int[] lval) {
	    // first estimate ndata
	    ndata = 0;
	    System.out.println("atlas size: "+nx+" x "+ny+" x "+nz+" ("+nxyz+")");
	    
	    if (map2target!=null) {
	        System.out.println("image size: "+ntx+" x "+nty+" x "+ntz+" ("+ntxyz+")");
            for (int x=0;x<ntx;x++) for (int y=0;y<nty;y++) for (int z=0;z<ntz;z++) {
                int idx = x+ntx*y+ntx*nty*z;
                int xyz = Numerics.bounded(Numerics.round(map2target[idx]),0,nx-1)
                        + nx*Numerics.bounded(Numerics.round(map2target[idx+ntxyz]),0,ny-1)
                        + nx*ny*Numerics.bounded(Numerics.round(map2target[idx+2*ntxyz]),0,nz-1);
                if (lval[xyz]>0) ndata++;
            }
            System.out.println("work region size: "+ndata);
            spatialProbas = new float[nbest][ndata];
            spatialLabels = new int[nbest][ndata];
            mask = new boolean[ntxyz]; 
            int id=0;
            for (int x=0;x<ntx;x++) for (int y=0;y<nty;y++) for (int z=0;z<ntz;z++) {
                int idx = x+ntx*y+ntx*nty*z;
                int xyz = Numerics.bounded(Numerics.round(map2target[idx]),0,nx-1)
                        + nx*Numerics.bounded(Numerics.round(map2target[idx+ntxyz]),0,ny-1)
                        + nx*ny*Numerics.bounded(Numerics.round(map2target[idx+2*ntxyz]),0,nz-1);
                if (lval[xyz]>0) {
                    mask[idx] = true;
                    for (int best=0;best<nbest;best++) {
                        spatialProbas[best][id] = pval[xyz+best*nxyz];
                        spatialLabels[best][id] = lval[xyz+best*nxyz];
                    }
                    id++;
                } else {
                    mask[idx] = false;
                }
            }
            nx = ntx; ny = nty; nz = ntz; nxyz = ntxyz;
            map2target = null;
            map2atlas = null;
	    } else {
            for (int xyz=0;xyz<nxyz;xyz++) if (lval[xyz]>0) ndata++;
            System.out.println("work region size: "+ndata);
            spatialProbas = new float[nbest][ndata];
            spatialLabels = new int[nbest][ndata];
            mask = new boolean[nxyz]; 
            int id=0;
            for (int xyz=0;xyz<nxyz;xyz++) {
                if (lval[xyz]>0) {
                    mask[xyz] = true;
                    for (int best=0;best<nbest;best++) {
                        spatialProbas[best][id] = pval[xyz+best*nxyz];
                        spatialLabels[best][id] = lval[xyz+best*nxyz];
                    }
                    id++;
                } else {
                    mask[xyz] = false;
                }
            }
        }
	}
	public final void setConditionalMeanAndStdv(float[] mean, float[] stdv) {
	    condmean = new double[nc][nobj][nobj];
	    condstdv = new double[nc][nobj][nobj];
  		condpair = new boolean[nc][nobj][nobj];
        for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) for (int c=0;c<nc;c++) {
	        condmean[c][obj1][obj2] = mean[obj1+obj2*nobj+c*nobj*nobj];
	        condstdv[c][obj1][obj2] = mean[obj1+obj2*nobj+c*nobj*nobj];
	        if (condstdv[c][obj1][obj2]>0) condpair[c][obj1][obj2] = true;
	        else condpair[c][obj1][obj2] = false;
	    }
	}

	public final void setConditionalHistogram(float[] val) {
	    //nbins = n;
	    condhistogram = new double[nc][nobj][nobj][nbins];
	    condmin = new double[nc][nobj][nobj];
	    condmax = new double[nc][nobj][nobj];
		condpair = new boolean[nc][nobj][nobj];
		objVolumeMean = new float[nobj];
		objVolumeStdv = new float[nobj];
	    for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) for (int c=0;c<nc;c++) {
	        condpair[c][obj1][obj2] = false;
	        condmin[c][obj1][obj2] = val[obj2+obj1*nobj+nobj*nobj*0+nobj*nobj*(nbins+4)*c];
	        for (int bin=0;bin<nbins;bin++) {
	            condhistogram[c][obj1][obj2][bin] = val[obj2+obj1*nobj+nobj*nobj*(bin+1)+nobj*nobj*(nbins+4)*c];
	            if (condhistogram[c][obj1][obj2][bin]>0) condpair[c][obj1][obj2] = true;
	        }
	        condmax[c][obj1][obj2] = val[obj2+obj1*nobj+nobj*nobj*(nbins+1)+nobj*nobj*(nbins+4)*c];
	        if (obj1==obj2) {
	            objVolumeMean[obj1] = val[obj2+obj1*nobj+nobj*nobj*(nbins+2)+nobj*nobj*(nbins+4)*c];
	            objVolumeStdv[obj1] = val[obj2+obj1*nobj+nobj*nobj*(nbins+3)+nobj*nobj*(nbins+4)*c];
	        }
	    }
	}
	
	public final void setOptions(boolean mB, boolean cB, boolean cA, boolean sP, boolean mP) {
	    modelBackground = mB;
	    cancelBackground = cB;
	    cancelAll = cA;
	    sumPosterior = sP;
	    maxPosterior = mP;
	    if (modelBackground) nobj = nobj+1;
	}
	
	public final void setDiffusionParameters(int iter, float diff) {
	    maxiter = iter;
	    maxdiff = diff;
	}
	
	public final void setHistogramModeling(boolean val) {
	    modelHistogram = val;
	}
	
	//public static final void setFollowSkeleton(boolean val) { skelParam=val; }
	//public final void setCorrectSkeletonTopology(boolean val) { topoParam=val; }
	//public final void setTopologyLUTdirectory(String val) { lutdir = val; }

	public final void setAtlasDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setAtlasDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setAtlasResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setAtlasResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
	
	public final void setTargetDimensions(int x, int y, int z) { ntx=x; nty=y; ntz=z; ntxyz=ntx*nty*ntz; }
	public final void setTargetDimensions(int[] dim) { ntx=dim[0]; nty=dim[1]; ntz=dim[2]; ntxyz=ntx*nty*ntz; }
	
	public final void setTargetResolutions(float x, float y, float z) { rtx=x; rty=y; rtz=z; }
	public final void setTargetResolutions(float[] res) { rtx=res[0]; rty=res[1]; rtz=res[2]; }
	
	// to be used for JIST definitions, generic info / help
	public final String getPackage() { return "IMCN Toolkit"; }
	public final String getCategory() { return "Segmentation"; }
	public final String getLabel() { return "Conditional Shape Segmentation"; }
	public final String getName() { return "ConditionalShapeSegmentation"; }

	public final String[] getAlgorithmAuthors() { return new String[]{"Pierre-Louis Bazin"}; }
	public final String getAffiliation() { return "Integrative Model-basec Cognitve Neuroscience unit, Universiteit van Amsterdam | Max Planck Institute for Human Cognitive and Brain Sciences"; }
	public final String getDescription() { return "Combines a collection of levelset surfaces and intensity maps into a condfitional segmentation"; }
	public final String getLongDescription() { return getDescription(); }
		
	public final String getVersion() { return "1.0"; };

	// create outputs
	public final int getBestDimension() { return nbest; }
	
	public final float[] getBestSpatialProbabilityMaps(int nval) {
	    nval = Numerics.min(nval,nbest);
        float[] images = new float[nval*nxyz];
		int id=0;
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int best=0;best<nval;best++) {
                images[xyz+best*nxyz] = spatialProbas[best][id];
            }
            id++;
        }
        return images;
	}
	public final int[] getBestSpatialProbabilityLabels(int nval) {
        nval = Numerics.min(nval,nbest);
        int[] images = new int[nval*nxyz];
		int id=0;
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int best=0;best<nval;best++) {
                images[xyz+best*nxyz] = spatialLabels[best][id];
            }
            id++;
        }
        return images;
	}

	public final float[] getBestIntensityProbabilityMaps(int nval) {
        nval = Numerics.min(nval,nbest);
        float[] images = new float[nval*nxyz];
		int id=0;
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int best=0;best<nval;best++) {
                images[xyz+best*nxyz] = intensityProbas[best][id];
            }
            id++;
        }
        return images;
	}
	public final int[] getBestIntensityProbabilityLabels(int nval) {
        nval = Numerics.min(nval,nbest);
        int[] images = new int[nval*nxyz];
		int id=0;
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int best=0;best<nval;best++) {
                images[xyz+best*nxyz] = intensityLabels[best][id];
            }
            id++;
        }
        return images;
	}

    public final float[] getBestProbabilityMaps(int nval) { 
        nval = Numerics.min(nval,nbest);
        float[] images = new float[nval*nxyz];
		int id=0;
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int best=0;best<nval;best++) {
                images[xyz+best*nxyz] = combinedProbas[best][id];
            }
            id++;
        }
        return images;
	}
	public final int[] getBestProbabilityLabels(int nval) { 
        nval = Numerics.min(nval,nbest);
        int[] images = new int[nval*nxyz];
		int id=0;
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int best=0;best<nval;best++) {
                images[xyz+best*nxyz] = combinedLabels[best][id];
            }
            id++;
        }
        return images;
	}
	
	public final float[] getCertaintyProbability() { 
        float[] images = new float[nxyz];
		int id=0;
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    images[xyz] = combinedProbas[0][id]-combinedProbas[1][id];
            id++;
        }
        return images;
	}
    public final float[] getNeighborhoodMaps(int nval) { 
        float[] images = new float[nval*nxyz];
		int id=0;
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int best=0;best<nval;best++) {
                images[xyz+best*nxyz] = ngbw[best][id];
            }
            id++;
        }
        return images;
	}
	public final int[] getFinalLabel() { return finalLabel; }
	
	public final float[] getFinalProba() { return finalProba; }
	
	public final float[] getConditionalMean() {
	    float[] val = new float[nc*nobj*nobj];
	    for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) for (int c=0;c<nc;c++) {
	        val[obj1+obj2*nobj+c*nobj*nobj] = (float)condmean[c][obj1][obj2];
	    }
	    return val;
	}
	
	public final float[] getConditionalStdv() {
	    float[] val = new float[nc*nobj*nobj];
	    for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) for (int c=0;c<nc;c++) {
	        val[obj1+obj2*nobj+c*nobj*nobj] = (float)condstdv[c][obj1][obj2];
	    }
	    return val;
	}
	
	public final float[] getConditionalHistogram() {
	    float[] val = new float[nc*nobj*nobj*(nbins+4)];
	    for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) for (int c=0;c<nc;c++) {
	        val[obj2+obj1*nobj+0*nobj*nobj+c*nobj*nobj*(nbins+4)] = (float)condmin[c][obj1][obj2];
	        for (int bin=0;bin<nbins;bin++) {
	            val[obj2+obj1*nobj+(bin+1)*nobj*nobj+c*nobj*nobj*(nbins+4)] = (float)condhistogram[c][obj1][obj2][bin];
	        }
	        val[obj2+obj1*nobj+(nbins+1)*nobj*nobj+c*nobj*nobj*(nbins+4)] = (float)condmax[c][obj1][obj2];
	        if (obj1==obj2) {
	            val[obj2+obj1*nobj+(nbins+2)*nobj*nobj+c*nobj*nobj*(nbins+4)] = (float)objVolumeMean[obj1];
	            val[obj2+obj1*nobj+(nbins+3)*nobj*nobj+c*nobj*nobj*(nbins+4)] = (float)objVolumeStdv[obj1];
	        }
	    }
	    return val;
	}
	
	public final int getNumberOfBins() {
	    return nbins;
	}
	
	public void execute() {
	    
	    System.out.println("dimensions: "+nsub+" subjects, "+nc+" contrasts, "+nobj+" objects");
	
	    if (spatialProbas==null || spatialLabels==null) {
	        computeAtlasPriors();
	    }
	    estimateTarget();
	    strictSimilarityDiffusion(6);
	    collapseConditionalMaps();
    }	    
	    
	public final void computeAtlasPriors() {
	    float[][][] levelsets = null; 
	    
	    // not correct: explicitly build the levelset of the background first, then crop it
	    if (modelBackground) {
            // adding the background: building a ring around the structures of interest
            // with also a sharp decay to the boundary
            levelsets = new float[nsub][nobj][];
            float[] background = new float[nxyz];
            boolean[] bgmask = new boolean[nxyz];
            for (int xyz=0;xyz<nxyz;xyz++) bgmask[xyz] = true;
 
            for (int sub=0;sub<nsub;sub++) {
                for (int xyz=0;xyz<nxyz;xyz++) {
                    float mindist = boundary;
                    for (int obj=0;obj<nobj-1;obj++) {
                        if (lvlImages[sub][obj][xyz]<mindist) mindist = lvlImages[sub][obj][xyz];
                    }
                    if (mindist<boundary/2.0) {
                        background[xyz] = -mindist;
                    } else {
                        background[xyz] = -boundary/2.0f + (mindist-boundary/2.0f);
                    }
                }
                InflateGdm gdm = new InflateGdm(background, nx, ny, nz, rx, ry, rz, bgmask, 0.4f, 0.4f, "no", null);
                gdm.evolveNarrowBand(0, 1.0f);
                levelsets[sub][0] = gdm.getLevelSet();
                for (int obj=1;obj<nobj;obj++) {
                    levelsets[sub][obj] = lvlImages[sub][obj-1];
                }
            }
            //nobj = nobj+1;
        } else {
            levelsets = lvlImages;
		}
		// mask anything too far outside the structures of interest
		mask = new boolean[nxyz];
		ndata = 0;
		for (int xyz=0;xyz<nxyz;xyz++) {
		    float mindist = boundary;
            for (int sub=0;sub<nsub;sub++) for (int obj=0;obj<nobj;obj++) {
                if (levelsets[sub][obj][xyz]<mindist) mindist = levelsets[sub][obj][xyz];
            }
            if (mindist<boundary) {
                mask[xyz] = true;
                ndata++;
            } else {
                mask[xyz] = false;
            }
        }
        System.out.println("masking: work region "+ndata+", compression: "+(ndata/(float)nxyz));
		
        // adapt number of kept values?
        
		System.out.println("compute joint conditional shape priors");
		spatialProbas = new float[nbest][ndata]; 
		spatialLabels = new int[nbest][ndata];
		
		int ctr = Numerics.floor(nsub/2);
        int dev = Numerics.floor(nsub/4);
                    
		double[] val = new double[nsub];
		int id=0;
		//double iqrsum=0, iqrden=0;
		double stdsum=0, stdden=0;
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    double[][] priors = new double[nobj][nobj];
            for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                //priors[obj1][obj2] = FastMath.exp( -0.5*med*med/(1.349*iqr*1.349*iqr) );
                // alternative idea: use a combination of mean and stdev as distance basis
                // -> take into account uncertainty better
                double mean = 0.0;
                for (int sub=0;sub<nsub;sub++) {
                    mean += Numerics.max(0.0, levelsets[sub][obj1][xyz]-deltaOut, levelsets[sub][obj2][xyz]-deltaIn);
                }
                mean /= nsub;
                double var = 0.0;
                for (int sub=0;sub<nsub;sub++) {
                    var += Numerics.square(mean-Numerics.max(0.0, levelsets[sub][obj1][xyz]-deltaOut, levelsets[sub][obj2][xyz]-deltaIn));
                }
                var = FastMath.sqrt(var/nsub);
                
                stdsum += var;
                stdden ++;
                
                double sigma2 = var+Numerics.max(deltaOut, deltaIn, 1.0);
                sigma2 *= sigma2;
                priors[obj1][obj2] = 1.0/FastMath.sqrt(2.0*FastMath.PI*sigma2)*FastMath.exp( -0.5*mean*mean/sigma2 );
 			}
            for (int best=0;best<nbest;best++) {
                int best1=0;
				int best2=0;
					
                for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                    if (priors[obj1][obj2]>priors[best1][best2]) {
						best1 = obj1;
						best2 = obj2;
					}
				}
				// check if best is zero: give null label in that case
				if (priors[best1][best2]>0) {
                    // sub optimal labeling, but easy to read
                    spatialLabels[best][id] = 100*(best1+1)+(best2+1);
                    spatialProbas[best][id] = (float)priors[best1][best2];
                } else {
                    for (int b=best;b<nbest;b++) {
                        spatialLabels[b][id] = 0;
                        spatialProbas[b][id] = 0.0f;
                    }
                    best = nbest;
                }                    
                // remove best value
                priors[best1][best2] = 0.0;
 		    }
 		    id++;
		}
		//System.out.println("mean spatial iqr: "+(iqrsum/iqrden));
		System.out.println("mean spatial stdev: "+(stdsum/stdden));
		// levelsets are now discarded...
		// not yet! 
		//levelsets = null;
		
		// rescale top % in each shape and intensity priors
		if (rescaleProbas){
            Percentile measure = new Percentile();
            val = new double[ndata];
            for (id=0;id<ndata;id++) val[id] = spatialProbas[0][id];
            float shapeMax = (float)measure.evaluate(val, top);
            System.out.println("top "+top+"% shape probability: "+shapeMax);
            for (id=0;id<ndata;id++) for (int best=0;best<nbest;best++) {
                spatialProbas[best][id] = (float)Numerics.min(top/100.0*spatialProbas[best][id]/shapeMax, 1.0f);
            }		
		}
		
		System.out.println("compute joint conditional intensity priors");
		
		float[][][] contrasts = intensImages;
		float[][] medc = new float[nc][ndata];
		float[][] iqrc = new float[nc][ndata];
		
		System.out.println("1. estimate subjects distribution");
		double[] cntsum = new double[nc];
		double[] cntden = new double[nc];
		val = new double[nsub];
		id = 0;
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int c=0;c<nc;c++) {
                for  (int sub=0;sub<nsub;sub++) {
                    val[sub] = contrasts[sub][c][xyz];
                }
                /*
                //System.out.println("values");
                Percentile measure = new Percentile();
                measure.setData(val);
			
                medc[c][id] = (float)measure.evaluate(50.0); 
                //System.out.println("median "+medc[c][id]);
                iqrc[c][id] = (float)(measure.evaluate(75.0) - measure.evaluate(25.0));
                //System.out.println("iqr "+iqrc[c][id]);
                */
                Numerics.sort(val);
                double med, iqr;
                if (nsub%2==0) {
                    med = 0.5*(val[ctr-1]+val[ctr]);
                    iqr = val[ctr+dev] - val[ctr-1-dev];
                } else {
                    med = val[ctr];
                    iqr = val[ctr+dev] - val[ctr-dev];
                }                   
                medc[c][id] = (float)med;
                iqrc[c][id] = (float)iqr;
                
                cntsum[c] += iqr;
                cntden[c]++;
            }
            id++;
        }
        for (int c=0;c<nc;c++) {
            System.out.println("mean iqr (contrast "+c+"): "+(cntsum[c]/cntden[c]));
		}
		
		System.out.println("2. compute conditional maps");
		
		condpair = new boolean[nc][nobj][nobj];
		if (modelHistogram) {
		    System.out.println("(use histograms for intensities)");
		    condmin = new double[nc][nobj][nobj];
		    condmax = new double[nc][nobj][nobj];
		    condhistogram = new double[nc][nobj][nobj][nbins];
		    
		    // min, max: percentile on median (to avoid spreading the values to outliers)
		    // same min, max for all object pairs => needed for fair comparison 
		    // (or normalize by volume, i.e. taking width into account
		    for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
		        System.out.print("\n("+(obj1+1)+" | "+(obj2+1)+"): ");
                for (int c=0;c<nc;c++) {
                    condmin[c][obj1][obj2] = 1e9f;
                    condmax[c][obj1][obj2] = -1e9f;
                }
                for (int c=0;c<nc;c++) {
                    boolean existsPair = false;
                    // use median intensities to estimate [min,max]
                    id = 0;
                    for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
                        double med = medc[c][id];
                        double iqr = iqrc[c][id];
                        // assuming here that iqr==0 means masked regions
                        if (iqr>0) { 
                            // look only among non-zero priors for each region
                            for (int best=0;best<nbest;best++) {
                                if (spatialLabels[best][id]==100*(obj1+1)+(obj2+1)) {
                                    // found value: proceeed
                                    if (med<condmin[c][obj1][obj2]) condmin[c][obj1][obj2] = med;
                                    if (med>condmax[c][obj1][obj2]) condmax[c][obj1][obj2] = med;
                                    if (condmin[c][obj1][obj2]!=condmax[c][obj1][obj2]) existsPair = true;
                                }
                            }
                        }
                        id++;
                    }
                    if (existsPair) {
                        condpair[c][obj1][obj2] = true;
                        System.out.print("["+condmin[c][obj1][obj2]+" , "+condmax[c][obj1][obj2]+"]    ");
                    } else {
                        condmin[c][obj1][obj2] = 0;
                        condmax[c][obj1][obj2] = 0;
                        condpair[c][obj1][obj2] = false;
                        System.out.print("empty pair    ");
                    }
                }
            }
            // take the global min,max to make histograms comparable
		    for (int c=0;c<nc;c++) {
		        double cmin = condmin[c][0][0];
		        double cmax = condmax[c][0][0];
		        for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                    if (condpair[c][obj1][obj2]) {
                        if (condmin[c][obj1][obj2]<cmin) cmin = condmin[c][obj1][obj2];
                        if (condmax[c][obj1][obj2]>cmax) cmax = condmax[c][obj1][obj2];
                    }
                }
		        for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
		            condmin[c][obj1][obj2] = cmin;
		            condmax[c][obj1][obj2] = cmax;
		        }
		    }
		    for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
		        for (int c=0;c<nc;c++) {
		            if (condpair[c][obj1][obj2]) {
                        id = 0;
                        for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
                            double med = medc[c][id];
                            double iqr = iqrc[c][id];
                            // assuming here that iqr==0 means masked regions
                            if (iqr>0) { 
                                // look for non-zero priors
                                for (int best=0;best<nbest;best++) {
                                    if (spatialLabels[best][id]==100*(obj1+1)+(obj2+1)) {
                                        // found value: proceeed
                                        for (int sub=0;sub<nsub;sub++) {
                                            // adds uncertainties from mismatch between subject intensities and mean shape
                                            /*
                                            double psub = spatialProbas[best][id]*1.0/FastMath.sqrt(2.0*FastMath.PI*1.349*iqr*1.349*iqr)
                                                               *FastMath.exp( -0.5*(contrasts[sub][c][xyz]-med)*(contrasts[sub][c][xyz]-med)/(1.349*iqr*1.349*iqr) );
                                            */
                                            double ldist = Numerics.max(levelsets[sub][obj1][xyz]-deltaOut, levelsets[sub][obj2][xyz]-deltaIn, 0.0);
                                            double ldelta = Numerics.max(deltaOut, deltaIn, 1.0);
                                            double pshape = FastMath.exp(-0.5*(ldist*ldist)/(ldelta*ldelta));
                                            double psub = pshape*1.0/FastMath.sqrt(2.0*FastMath.PI*1.349*iqr*1.349*iqr)
                                                               *FastMath.exp( -0.5*(contrasts[sub][c][xyz]-med)*(contrasts[sub][c][xyz]-med)/(1.349*iqr*1.349*iqr) );
                                            // add to the mean
                                            int bin = Numerics.bounded(Numerics.ceil( (contrasts[sub][c][xyz]-condmin[c][obj1][obj2])/(condmax[c][obj1][obj2]-condmin[c][obj1][obj2])*nbins)-1, 0, nbins-1);
                                            condhistogram[c][obj1][obj2][bin] += psub;
                                        }
                                        best=nbest;
                                    }
                                }
                            }
                            id++;
                        }
                        // smooth histograms to avoid sharp edge effects
                        double var = 1.0*1.0;
                        double[] tmphist = new double[nbins];
                        for (int bin1=0;bin1<nbins;bin1++) {
                            for (int bin2=0;bin2<nbins;bin2++) {
                                tmphist[bin1] += condhistogram[c][obj1][obj2][bin2]*FastMath.exp(-0.5*(bin1-bin2)*(bin1-bin2)/var);
                            }
                        }
                        for (int bin=0;bin<nbins;bin++) condhistogram[c][obj1][obj2][bin] = tmphist[bin];
                        
                        // normalize: sum over count x spread = 1
                        double sum = 0.0;
                        for (int bin=0;bin<nbins;bin++) sum += condhistogram[c][obj1][obj2][bin];   
                        //for (int bin=0;bin<nbins;bin++) condhistogram[c][obj1][obj2][bin] /= sum*(condmax[c][obj1][obj2]-condmin[c][obj1][obj2]);   
                        for (int bin=0;bin<nbins;bin++) condhistogram[c][obj1][obj2][bin] /= sum;   
                    } else {
                        for (int bin=0;bin<nbins;bin++) condhistogram[c][obj1][obj2][bin] = 0;   
                    }
                }
            }
        } else {
            // use spatial priors and subject variability priors to define conditional intensity
            // mean and stdev
            System.out.println("(use mean,stdev for intensities)");
            condmean = new double[nc][nobj][nobj];
            condstdv = new double[nc][nobj][nobj];
            for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                System.out.print("\n("+(obj1+1)+" | "+(obj2+1)+"): ");
                for (int c=0;c<nc;c++) {
                   // System.out.println("..mean");
                   double sum = 0.0;
                   double den = 0.0;
                   id = 0;
                   for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
                       double med = medc[c][id];
                       double iqr = iqrc[c][id];
                       // assuming here that iqr==0 means masked regions
                       if (iqr>0) { 
                           // look for non-zero priors
                           for (int best=0;best<nbest;best++) {
                               if (spatialLabels[best][id]==100*(obj1+1)+(obj2+1)) {
                                   // found value: proceeed
                                   for (int sub=0;sub<nsub;sub++) {
                                       // adds uncertainties from mismatch between subject intensities and mean shape
                                       double ldist = Numerics.max(levelsets[sub][obj1][xyz]-deltaOut, levelsets[sub][obj2][xyz]-deltaIn, 0.0);
                                       double ldelta = Numerics.max(deltaOut, deltaIn, 1.0);
                                       double pshape = FastMath.exp(-0.5*(ldist*ldist)/(ldelta*ldelta));
                                       double psub = pshape*1.0/FastMath.sqrt(2.0*FastMath.PI*1.349*iqr*1.349*iqr)
                                                          *FastMath.exp( -0.5*(contrasts[sub][c][xyz]-med)*(contrasts[sub][c][xyz]-med)/(1.349*iqr*1.349*iqr) );
                                       // add to the mean
                                       sum += psub*contrasts[sub][c][xyz];
                                       den += psub;
                                   }
                                   best=nbest;
                               }
                           }
                       }
                       id++;
                   }
                   // build average
                   if (den>0) {
                       condmean[c][obj1][obj2] = sum/den;
                       condpair[c][obj1][obj2] = true;
                   } else {
                       System.out.print("empty pair        ");
                       condmean[c][obj1][obj2] = 0.0;
                       condpair[c][obj1][obj2] = false;
                   }
                   //System.out.println("..stdev");
                   double var = 0.0;
                   id = 0;
                   for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
                       double med = medc[c][id];
                       double iqr = iqrc[c][id];
                       // assuming here that iqr==0 means masked regions
                       if (iqr>0) { 
                           // look for non-zero priors
                           for (int best=0;best<nbest;best++) {
                               if (spatialLabels[best][id]==100*(obj1+1)+(obj2+1)) {
                                   // found value: proceeed
                                   for (int sub=0;sub<nsub;sub++) {
                                       // adds uncertainties from mismatch between subject intensities and mean shape
                                       double ldist = Numerics.max(levelsets[sub][obj1][xyz]-deltaOut, levelsets[sub][obj2][xyz]-deltaIn, 0.0);
                                       double ldelta = Numerics.max(deltaOut, deltaIn, 1.0);
                                       double pshape = FastMath.exp(-0.5*(ldist*ldist)/(ldelta*ldelta));
                                       double psub = pshape*1.0/FastMath.sqrt(2.0*FastMath.PI*1.349*iqr*1.349*iqr)
                                                           *FastMath.exp( -0.5*(contrasts[sub][c][xyz]-med)*(contrasts[sub][c][xyz]-med)/(1.349*iqr*1.349*iqr) );
                                       // add to the mean
                                       var += psub*(contrasts[sub][c][xyz]-condmean[c][obj1][obj2])*(contrasts[sub][c][xyz]-condmean[c][obj1][obj2]);
                                   }
                                   best=nbest;
                               }
                           }
                       }
                       id++;
                   }
                   // build stdev
                   if (var==0) {
                       System.out.print("empty region        ");
                       condstdv[c][obj1][obj2] = 0;
                       condpair[c][obj1][obj2] = false;
                   } else if (den>0) {
                       condstdv[c][obj1][obj2] = FastMath.sqrt(var/den);
                       System.out.print(condmean[c][obj1][obj2]+" +/- "+condstdv[c][obj1][obj2]+"    ");
                       condpair[c][obj1][obj2] = true;
                   } else {
                       System.out.print("empty pair        ");
                       condstdv[c][obj1][obj2] = 0;
                       condpair[c][obj1][obj2] = false;
                   } 
                }
            }
        }
        // compute volume mean, stdv of each structure
        objVolumeMean = new float[nobj];
        objVolumeStdv = new float[nobj];
        for (int obj=0;obj<nobj;obj++) {
            float[] vols = new float[nsub];
            for (int sub=0;sub<nsub;sub++) {
                vols[sub] = 0.0f;
                for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
                    if (levelsets[sub][obj][xyz]<0) {
                        vols[sub]+=rx*ry*rz;
                    }
                }
                objVolumeMean[obj] += vols[sub]/nsub;
            }
            for (int sub=0;sub<nsub;sub++) {
                objVolumeStdv[obj] += Numerics.square(vols[sub]-objVolumeMean[obj])/(nsub-1.0f);
            }
            objVolumeStdv[obj] = (float)FastMath.sqrt(objVolumeStdv[obj]);
        }        
        
		// at this point the atlas data is not used anymore
		levelsets = null;
		contrasts = null;
		System.out.println("\ndone");
	}

	public final void estimateTarget() {	
		
		// compute the median of stdevs from atlas -> scale for image distances
		// use only the j|j labels -> intra class variations
		double[] stdevs = new double[nobj];
		medstdv= new float[nc];
		for (int c=0;c<nc;c++) {
		    int ndev=0;
		    for (int obj=0;obj<nobj;obj++) {
		        if (condpair[c][obj][obj]) {
		            if (modelHistogram) {
                        double sum = 0.0;
                        double den = 0.0;
                        for (int bin=0;bin<nbins;bin++) {
                            sum += condhistogram[c][obj][obj][bin]*(condmin[c][obj][obj]+bin*(condmax[c][obj][obj]-condmin[c][obj][obj])/nbins);
                            den += condhistogram[c][obj][obj][bin];
                        }
                        sum /= den;
                        double var = 0.0;
                        for (int bin=0;bin<nbins;bin++) {
                            double val = condmin[c][obj][obj]+bin*(condmax[c][obj][obj]-condmin[c][obj][obj])/nbins;
                            var += condhistogram[c][obj][obj][bin]*(val-sum)*(val-sum);
                        }
                        stdevs[ndev] = FastMath.sqrt(var/den);
                    } else {
                        stdevs[ndev] = condstdv[c][obj][obj];
                    }
                    ndev++;
                }
		    }
		    Percentile measure = new Percentile();
            medstdv[c] = (float)measure.evaluate(stdevs, 0, ndev, 50.0);
        }
        stdevs = null;

        for (int c=0;c<nc;c++) {
            System.out.println("median intra-class stdev (contrast "+c+"): "+medstdv[c]);
		}
        
        System.out.println("apply priors to target");
        
        float[][][] separateIntensProbas = new float[nc][nbest][ndata]; 
		int[][][] separateIntensLabels = new int[nc][nbest][ndata];
			
		// if target in a different space, resample
		float[][] target = null;
		if (map2atlas!=null) {
		    target = new float[nc][nxyz];
		    for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		        int idx = Numerics.bounded(Numerics.round(map2atlas[xyz]),0,ntx-1)
		                + ntx*Numerics.bounded(Numerics.round(map2atlas[xyz+nxyz]),0,nty-1)
		                + ntx*nty*Numerics.bounded(Numerics.round(map2atlas[xyz+2*nxyz]),0,ntz-1);
		        for (int c=0;c<nc;c++) target[c][xyz] = targetImages[c][idx];
		    }
		    // replace original data for further processing
		    targetImages = target;
		} else {
		    target = targetImages;
        }
		
		// combine priors and contrasts posteriors (update the priors maps)
		for (int c=0;c<nc;c++) {
            int id=0;
            for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
               double[][] likelihood = new double[nobj][nobj];
               for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                   // look for non-zero priors
                   likelihood[obj1][obj2] = 0.0;
                   
                   // impose the leftout classes here
                   if (cancelBackground && obj1==0 && obj2==0) {
                        likelihood[obj1][obj2] = 0.0;
                    } else if (cancelAll && obj1==obj2) {
                        likelihood[obj1][obj2] = 0.0;
                    } else {                   
                       for (int best=0;best<nbest;best++) {
                           if (spatialLabels[best][id]==100*(obj1+1)+(obj2+1)) {
                               // multiply nc times to balance prior and posterior
                               //likelihood[obj1][obj2] = 1.0;
                               likelihood[obj1][obj2] = spatialProbas[best][id];
                               best = nbest;
                           }
                       }
                    }
                   if (likelihood[obj1][obj2]>0) {
                       if (condpair[c][obj1][obj2]) {
                           double pobjc;
                           if (modelHistogram) {
                               int bin = Numerics.bounded(Numerics.ceil( (target[c][xyz]-condmin[c][obj1][obj2])/(condmax[c][obj1][obj2]-condmin[c][obj1][obj2])*nbins)-1, 0, nbins-1);
                               pobjc = medstdv[c]*condhistogram[c][obj1][obj2][bin];
                               //pobjc = condhistogram[c][obj1][obj2][bin];
                           } else {
                               pobjc = medstdv[c]/FastMath.sqrt(2.0*FastMath.PI*condstdv[c][obj1][obj2]*condstdv[c][obj1][obj2])
                                            *FastMath.exp( -0.5*(target[c][xyz]-condmean[c][obj1][obj2])*(target[c][xyz]-condmean[c][obj1][obj2])
                                                               /(condstdv[c][obj1][obj2]*condstdv[c][obj1][obj2]) );
                               //pobjc = 1.0/FastMath.sqrt(2.0*FastMath.PI*condstdv[c][obj1][obj2]*condstdv[c][obj1][obj2])
                               //             *FastMath.exp( -0.5*(target[c][xyz]-condmean[c][obj1][obj2])*(target[c][xyz]-condmean[c][obj1][obj2])
                               //                                /(condstdv[c][obj1][obj2]*condstdv[c][obj1][obj2]) );
                           }
                           likelihood[obj1][obj2] *= pobjc;
                       } else {
                           // what to do here? does it ever happen?
                           //System.out.print("!");
                       }
                   }
                }
                for (int best=0;best<nbest;best++) {
                    int best1=0;
                    int best2=0;
                        
                    for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                        if (likelihood[obj1][obj2]>likelihood[best1][best2]) {
                            best1 = obj1;
                            best2 = obj2;
                        }
                    }
                    // now find the corresponding shape prior
                    double shapeprior = 1.0;
                    for (int sbest=0;sbest<nbest;sbest++) {
                       if (spatialLabels[sbest][id]==100*(best1+1)+(best2+1)) {
                           shapeprior = spatialProbas[sbest][id];
                           sbest = nbest;
                       }
                    }
                    // sub optimal labeling, but easy to read
                    separateIntensLabels[c][best][id] = 100*(best1+1)+(best2+1);
                    separateIntensProbas[c][best][id] = (float)(likelihood[best1][best2]/shapeprior);
                    // remove best value
                    likelihood[best1][best2] = 0.0;
                }
                id++;
            }
            if (rescaleIntensities) {
                // rescale top % in each shape and intensity priors
                Percentile measure = new Percentile();
                double[] val = new double[ndata];
                for (id=0;id<ndata;id++) val[id] = separateIntensProbas[c][0][id];
                float intensMax = (float)measure.evaluate(val, top);
                System.out.println("top "+top+"% intensity probability (contrast "+c+"): "+intensMax);
                
                for (id=0;id<ndata;id++) for (int best=0;best<nbest;best++) {
                    separateIntensProbas[c][best][id] = (float)Numerics.min(top/100.0*separateIntensProbas[c][best][id]/intensMax, 1.0f);
                }
            }
		}
        // combine the multiple contrasts            
		System.out.println("combine intensity probabilities");
		intensityProbas = new float[nbest][ndata]; 
		intensityLabels = new int[nbest][ndata];
        int id=0;
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
            double[][] likelihood = new double[nobj][nobj];
            for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
               likelihood[obj1][obj2] = 1.0;
               for (int c=0;c<nc;c++) {
                   double val = 0.0;
                   for (int best=0;best<nbest;best++) {
                       if (separateIntensLabels[c][best][id]==100*(obj1+1)+(obj2+1)) {
                           val = separateIntensProbas[c][best][id];
                           best = nbest;
                       }
                   }
                   likelihood[obj1][obj2] *= val;
               }
            }
            for (int best=0;best<nbest;best++) {
                int best1=0;
                int best2=0;
                    
                for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                    if (likelihood[obj1][obj2]>likelihood[best1][best2]) {
                        best1 = obj1;
                        best2 = obj2;
                    }
                }
                // sub optimal labeling, but easy to read
                intensityLabels[best][id] = 100*(best1+1)+(best2+1);
                // scaling for multiplicative intensities
                intensityProbas[best][id] = (float)FastMath.pow(likelihood[best1][best2],1.0/nc);
                // remove best value
                likelihood[best1][best2] = 0.0;
            }
            id++;
        }
		if (!rescaleIntensities && rescaleProbas) {
            Percentile measure = new Percentile();
            double[] val = new double[ndata];
            for (id=0;id<ndata;id++) val[id] = intensityProbas[0][id];
            float intensMax = (float)measure.evaluate(val, top);
            System.out.println("top "+top+"% global intensity probability: "+intensMax);
            for (id=0;id<ndata;id++) for (int best=0;best<nbest;best++) {
                intensityProbas[best][id] = (float)Numerics.min(top/100.0*intensityProbas[best][id]/intensMax, 1.0f);
            }		
		}
		
		// posterior : merge both measures
		System.out.println("generate posteriors");
        combinedProbas = new float[nbest][ndata]; 
		combinedLabels = new int[nbest][ndata];
        id=0;
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    double[][] posteriors = new double[nobj][nobj];
		    for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                // look for non-zero priors
                posteriors[obj1][obj2] = 0.0;
                
                for (int best=0;best<nbest;best++) {
                    if (spatialLabels[best][id]==100*(obj1+1)+(obj2+1)) {
                        // multiply nc times to balance prior and posterior
                        posteriors[obj1][obj2] = spatialProbas[best][id];
                        best = nbest;
                    }
                }
                if (posteriors[obj1][obj2]>0) {
                    double intensPrior = 0.0;
                    for (int best=0;best<nbest;best++) {
                        if (intensityLabels[best][id]==100*(obj1+1)+(obj2+1)) {
                            intensPrior = intensityProbas[best][id];
                            best=nbest;
                        }
                    }
                    posteriors[obj1][obj2] *= intensPrior;
                }
            }
            for (int best=0;best<nbest;best++) {
                int best1=0;
                int best2=0;
                    
                for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                    if (posteriors[obj1][obj2]>posteriors[best1][best2]) {
                        best1 = obj1;
                        best2 = obj2;
                    }
                }
                // sub optimal labeling, but easy to read
                combinedLabels[best][id] = 100*(best1+1)+(best2+1);
                combinedProbas[best][id] = (float)FastMath.sqrt(posteriors[best1][best2]);
                // remove best value
                posteriors[best1][best2] = 0.0;
 		    }
 		    id++;
		}
		
	}
	
	public final void collapseConditionalMaps() {	    
        for (int id=0;id<ndata;id++) {
            double[][] posteriors = new double[nobj][nobj];
            for (int best=0;best<nbest;best++) {
                int obj1 = Numerics.floor(combinedLabels[best][id]/100)-1;
                int obj2 = combinedLabels[best][id]-(obj1+1)*100-1;
                posteriors[obj1][obj2] = combinedProbas[best][id];
            }
            if (sumPosterior) {
                for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) if (obj2!=obj1) {
                   posteriors[obj1][obj1] += posteriors[obj1][obj2];
                   posteriors[obj1][obj2] = 0.0;
                }
            } else if (maxPosterior) {
                for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) if (obj2!=obj1) {
                    posteriors[obj1][obj1] = Numerics.max(posteriors[obj1][obj1],posteriors[obj1][obj2]);
                    posteriors[obj1][obj2] = 0.0;
                }
            }
            for (int best=0;best<nbest;best++) {
                int best1=0;
                int best2=0;
                    
                for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                    if (posteriors[obj1][obj2]>posteriors[best1][best2]) {
                        best1 = obj1;
                        best2 = obj2;
                    }
                }
                // sub optimal labeling, but easy to read
                //combinedLabels[best][id] = 100*(best1+1)+(best2+1);
                combinedLabels[best][id] = best1;
                combinedProbas[best][id] = (float)posteriors[best1][best2];
                // remove best value
                posteriors[best1][best2] = 0.0;
 		    }
        }
    }

	public final void strictSimilarityDiffusion(int nngb) {	
		
		float[][] target = targetImages;
		// add a local diffusion step?
		System.out.print("Diffusion step: \n");
		
		// build ID map
		idmap = new int[nxyz];
		int id = 0;
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    idmap[xyz] = id;
		    id++;
		}

		// graph = N-most likely neihgbors (based on target intensity)
		System.out.print("Build similarity neighborhood\n");
 		ngbw = new float[nngb+1][ndata];
		int[][] ngbi = new int[nngb][ndata];
		float[] ngbsim = new float[26];
		
		for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
		    int xyz = x+nx*y+nx*ny*z;
		    if (mask[xyz]) {
		        for (byte d=0;d<26;d++) {
		            int ngb = Ngb.neighborIndex(d, xyz, nx,ny,nz);
		            if (mask[ngb]) {
		                ngbsim[d] = 1.0f;
		                for (int c=0;c<nc;c++) {
		                    //ngbsim[d] *= 1.0f/Numerics.max(1e-6,Numerics.abs(target[c][xyz]-target[c][ngb])/medstdv[c]);
                            ngbsim[d] *= (float)FastMath.exp( -0.5/nc*(target[c][xyz]-target[c][ngb])*(target[c][xyz]-target[c][ngb])
                                         /(medstdv[c]*medstdv[c]) );
                        }
                        //if (ngbsim[d]==0) System.out.print("!");
                    } else {
                        ngbsim[d] = 0.0f;
                    }
                }
                // choose the N best ones
                ngbw[nngb][idmap[xyz]] = 0.0f;
                for (int n=0;n<nngb;n++) {
                    byte best=0;
                        
                    for (byte d=0;d<26;d++)
                        if (ngbsim[d]>ngbsim[best]) 
                            best = d;
                    
                    ngbw[n][idmap[xyz]] = ngbsim[best];
                    ngbi[n][idmap[xyz]] = idmap[Ngb.neighborIndex(best, xyz, nx,ny,nz)];
                    ngbw[nngb][idmap[xyz]] += ngbsim[best];
                    
                    ngbsim[best] = 0.0f;
                }
                //if (ngbw[nngb][idmap[xyz]]==0) System.out.print("0");
            }
        }  
		System.out.print("\n");

		// diffusion only between i|j <-> i|j, not i|j <-> i|k, i|j <-> j|i
		
		float[][] diffusedProbas = new float[nbest][ndata]; 
		int[][] diffusedLabels = new int[nbest][ndata];
		
		// first copy the originals, then iterate on the copy?
		for (id=0;id<ndata;id++) {
		    for (int best=0;best<nbest;best++) {
                diffusedProbas[best][id] = combinedProbas[best][id];
                diffusedLabels[best][id] = combinedLabels[best][id];
            }
        }
		    		
		double[][] diffused = new double[nobj][nobj];
        for (int t=0;t<maxiter;t++) {
		    for (id=0;id<ndata;id++) if (ngbw[nngb][id]>0) {
                for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                    diffused[obj1][obj2] = 0.0;
                    
                    for (int best=0;best<nbest;best++) {
                        if (combinedLabels[best][id]==100*(obj1+1)+(obj2+1)) {
                            diffused[obj1][obj2] = combinedProbas[best][id];
                        }
                    }
                    if (diffused[obj1][obj2]>0) {
                        float den = ngbw[nngb][id];
                        diffused[obj1][obj2] *= den;
                        
                        for (int n=0;n<nngb;n++) {
                            int ngb = ngbi[n][id];
                            float ngbmax = 0.0f;
                            // max over neighbors ( -> stop at first found)
                            for (int best=0;best<nbest;best++) {
                                // variable transition weights? use frequencies from training?
                                float transw = 0.0f;
                                if (combinedLabels[best][ngb]==100*(obj1+1)+(obj2+1)) transw = 1.0f;
                                //else if (combinedLabels[best][ngb]>100*(obj1+1) &&  combinedLabels[best][ngb]<100*(obj1+2)) transw = 1.0f;
                                // note that obj1==obj2 is covered above, no need to check here
                                //else if (combinedLabels[best][ngb]==100*(obj2+1)+(obj1+1)) transw = 0.0f;
                                else transw = 0.0f;
                                
                                if (transw>0) {
                                    ngbmax = Numerics.max(ngbmax, transw*combinedProbas[best][ngb]);
                                    best = nbest;
                                }
                            }
                            //if (ngbmax==0) System.out.print("0");
                            diffused[obj1][obj2] += ngbw[n][id]*ngbmax;
                            den += ngbw[n][id];
                        }
                        diffused[obj1][obj2] /= den;
                    }
                }
                for (int best=0;best<nbest;best++) {
                    int best1=0;
                    int best2=0;
                        
                    for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                        if (diffused[obj1][obj2]>diffused[best1][best2]) {
                            best1 = obj1;
                            best2 = obj2;
                        }
                    }
                    // sub optimal labeling, but easy to read
                    diffusedLabels[best][id] = 100*(best1+1)+(best2+1);
                    diffusedProbas[best][id] = (float)diffused[best1][best2];
                    // remove best value
                    diffused[best1][best2] = 0.0;
                }
            }
            double diff = 0.0;
            for (id=0;id<ndata;id++) for (int best=0;best<nbest;best++) {
                if (combinedLabels[best][id] == diffusedLabels[best][id]) {
                    //diff += Numerics.abs(diffusedProbas[best][id]-combinedProbas[best][id]);
                    diff += 0.0;
                } else {
                    //diff += Numerics.abs(diffusedProbas[best][id]-combinedProbas[best][id]);
                    diff += 1.0;
                }
                combinedLabels[best][id] = diffusedLabels[best][id];
                combinedProbas[best][id] = diffusedProbas[best][id];
            }
            System.out.println("diffusion step "+t+": "+(diff/ndata));
            if (diff/ndata<maxdiff) t=maxiter;
		}

		target = null;
	}
	
	public void optimalVolumeThreshold(float spread, float scale, boolean certainty) {
	    // main idea: region growing from inside, until within volume prior
	    // and a big enough difference in "certainty" score?
	    
		// find appropriate threshold to have correct volume; should use a fast marching approach!
		BinaryHeap2D	heap = new BinaryHeap2D(nx*ny+ny*nz+nz*nx, BinaryHeap4D.MAXTREE);
		int[] labels = new int[ndata];
        int[] start = new int[nobj];
        float[] bestscore = new float[nobj];
		heap.reset();
		// important: skip first label as background (allows for unbounded growth)
        for (int obj=1;obj<nobj;obj++) {
		    // find highest scoring voxel as starting point
            for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
                int xyz=x+nx*y+nx*ny*z;
                if (mask[xyz]) {
                    int id = idmap[xyz];
                    if (combinedLabels[0][id]==obj) {
                        float score;
                        if (certainty) {
                            score = combinedProbas[0][id]-combinedProbas[1][id];
                        } else {
                            score = combinedProbas[0][id];
                        }
                        if (score>bestscore[obj]) {
                            bestscore[obj] = score;
                            start[obj] = xyz;
                        }
                    }
                }
            }
            heap.addValue(bestscore[obj],start[obj],(byte)obj);
        }
        float[] prev = new float[nobj];
        double[] vol = new double[nobj];
        double[] bestvol = new double[nobj];
        double[] bestproba = new double[nobj];
        while (heap.isNotEmpty()) {
            float score = heap.getFirst();
            int xyz = heap.getFirstId();
            byte obj = heap.getFirstState();
            heap.removeFirst();
            if (labels[idmap[xyz]]==0) {
                double volmean = objVolumeMean[obj];
                double volstdv = objVolumeStdv[obj];
                // compute the joint probability function
                double pvol = FastMath.exp(-0.5*(vol[obj]-volmean)*(vol[obj]-volmean)/(volstdv*volstdv));
                double pdiff = 1.0-FastMath.exp(-0.5*(score-prev[obj])*(score-prev[obj])/(scale*scale));
                
                double pstop = pvol*pdiff;
                if (pstop>bestproba[obj] && vol[obj]>volmean-spread*volstdv) {
                    bestproba[obj] = pstop;
                    bestvol[obj] = vol[obj];
                }
                // update the values
                vol[obj]+= rx*ry*rz;
                labels[idmap[xyz]] = obj;
                prev[obj] = score;
                
                // run until the volume exceeds the mean volume + n*stdev
                if (vol[obj]<volmean+spread*volstdv) {
                    // add neighbors
                    for (byte k = 0; k<6; k++) {
                        int ngb = Ngb.neighborIndex(k, xyz, nx, ny, nz);
                        if (mask[ngb]) {
                            if (labels[idmap[ngb]]==0) {
                                for (int best=0;best<nbest;best++) {
                                    if (combinedLabels[best][idmap[ngb]]==obj) {
                                        if (certainty) {
                                            if (best==0) {
                                                heap.addValue(combinedProbas[0][idmap[ngb]]-combinedProbas[1][idmap[ngb]],ngb,obj);
                                            } else {
                                                heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[0][idmap[ngb]],ngb,obj);
                                            }
                                        } else {
                                            heap.addValue(combinedProbas[best][idmap[ngb]],ngb,obj);
                                        }
                                        best=nbest;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        System.out.println("Average volumes: ");
        for (int obj=1;obj<nobj;obj++) System.out.println(obj+": "+objVolumeMean[obj]+" ("+objVolumeStdv[obj]+") ");
        System.out.println("\nOptimized volumes: ");
        for (int obj=1;obj<nobj;obj++) System.out.println(obj+": "+bestvol[obj]+" ("+bestproba[obj]+") ");
        // re-run one last time to get the segmentation
        heap.reset();
        for (int obj=0;obj<nobj;obj++) {
            vol[obj] = 0.0;
        }
        for(int id=0;id<ndata;id++) labels[id] = 0;
        for (int obj=1;obj<nobj;obj++) {
            heap.addValue(bestscore[obj],start[obj],(byte)obj);
        }
        while (heap.isNotEmpty()) {
            float score = heap.getFirst();
            int xyz = heap.getFirstId();
            byte obj = heap.getFirstState();
            heap.removeFirst();
            if (labels[idmap[xyz]]==0) {
                if (vol[obj]<bestvol[obj]) {
                    // update the values
                    vol[obj]+=rx*ry*rz;
                    labels[idmap[xyz]] = obj;
                
                    // add neighbors
                    for (byte k = 0; k<6; k++) {
                        int ngb = Ngb.neighborIndex(k, xyz, nx, ny, nz);
                        if (mask[ngb]) {
                            if (labels[idmap[ngb]]==0) {
                                for (int best=0;best<nbest;best++) {
                                    if (combinedLabels[best][idmap[ngb]]==obj) {
                                        if (certainty) {
                                            if (best==0) {
                                                heap.addValue(combinedProbas[0][idmap[ngb]]-combinedProbas[1][idmap[ngb]],ngb,obj);
                                            } else {
                                                heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[0][idmap[ngb]],ngb,obj);
                                            }
                                        } else {
                                            heap.addValue(combinedProbas[best][idmap[ngb]],ngb,obj);
                                        }
                                        best=nbest;
                                    }
                                }
                            }  
                        }
                    }
                }
            }
        }
        // final segmentation: collapse onto result images
        finalLabel = new int[nxyz];
        finalProba = new float[nxyz];
        float[] kept = new float[nobj];
        for (int x=1;x<ntx-1;x++) for (int y=1;y<nty-1;y++) for (int z=1;z<ntz-1;z++) {
            int xyz = x+ntx*y+ntx*nty*z;
            if (mask[xyz]) {
                int id = idmap[xyz];            
                int obj = labels[id];
                for (int best=0;best<nbest;best++) {
                    if (combinedLabels[best][id]==obj) {
                        if (certainty) {
                            if (best==0) {
                                finalProba[xyz] = combinedProbas[0][id]-combinedProbas[1][id];
                            } else {
                                finalProba[xyz] = combinedProbas[best][id]-combinedProbas[0][id];
                            }
                        } else {
                            finalProba[xyz] = combinedProbas[best][id];
                        }
                        best=nbest;
                    }
                }
                finalLabel[xyz] = obj;
            }
        }
        return;            
	}
	
	public void mappedOptimalVolumeThreshold(float spread, float scale, boolean certainty) {
	    // main idea: region growing from inside, until within volume prior
	    // and a big enough difference in "certainty" score?
	    
	    // using a coordinate mapping to output the result from atlas to subject space
	    
		// find appropriate threshold to have correct volume; should use a fast marching approach!
		BinaryHeap2D	heap = new BinaryHeap2D(nx*ny+ny*nz+nz*nx, BinaryHeap4D.MAXTREE);
		int[] labels = new int[ntxyz];
        int[] start = new int[nobj];
        float[] bestscore = new float[nobj];
		heap.reset();
		// important: skip first label as background (allows for unbounded growth)
        for (int obj=1;obj<nobj;obj++) {
		    // find highest scoring voxel as starting point
            for (int x=1;x<ntx-1;x++) for (int y=1;y<nty-1;y++) for (int z=1;z<ntz-1;z++) {
                int idx = x+ntx*y+ntx*nty*z;
                int xyz = Numerics.bounded(Numerics.round(map2target[idx]),0,nx-1)
                        + nx*Numerics.bounded(Numerics.round(map2target[idx+ntxyz]),0,ny-1)
                        + nx*ny*Numerics.bounded(Numerics.round(map2target[idx+2*ntxyz]),0,nz-1);
                if (mask[xyz]) {
                    int id = idmap[xyz];
                    if (combinedLabels[0][id]==obj) {
                        float score;
                        if (certainty) {
                            score = combinedProbas[0][id]-combinedProbas[1][id];
                        } else {
                            score = combinedProbas[0][id];
                        }
                        if (score>bestscore[obj]) {
                            bestscore[obj] = score;
                            start[obj] = idx;
                        }
                    }
                }
            }
            heap.addValue(bestscore[obj],start[obj],(byte)obj);
        }
        float[] prev = new float[nobj];
        double[] vol = new double[nobj];
        double[] bestvol = new double[nobj];
        double[] bestproba = new double[nobj];
        while (heap.isNotEmpty()) {
            float score = heap.getFirst();
            int idx = heap.getFirstId();
            int xyz = Numerics.bounded(Numerics.round(map2target[idx]),0,nx-1)
                    + nx*Numerics.bounded(Numerics.round(map2target[idx+ntxyz]),0,ny-1)
                    + nx*ny*Numerics.bounded(Numerics.round(map2target[idx+2*ntxyz]),0,nz-1);
            byte obj = heap.getFirstState();
            heap.removeFirst();
            if (labels[idx]==0) {
                double volmean = objVolumeMean[obj];
                double volstdv = objVolumeStdv[obj];
                // compute the joint probability function
                double pvol = FastMath.exp(-0.5*(vol[obj]-volmean)*(vol[obj]-volmean)/(volstdv*volstdv));
                double pdiff = 1.0-FastMath.exp(-0.5*(score-prev[obj])*(score-prev[obj])/(scale*scale));
                
                double pstop = pvol*pdiff;
                if (pstop>bestproba[obj] && vol[obj]>volmean-spread*volstdv) {
                    bestproba[obj] = pstop;
                    bestvol[obj] = vol[obj];
                }
                // update the values
                vol[obj] += rtx*rty*rtz;
                labels[idx] = obj;
                prev[obj] = score;
                
                // run until the volume exceeds the mean volume + n*stdev
                if (vol[obj]<volmean+spread*volstdv) {
                    // add neighbors
                    for (byte k = 0; k<6; k++) {
                        int idn = Ngb.neighborIndex(k, idx, ntx, nty, ntz);
                        int ngb = Numerics.bounded(Numerics.round(map2target[idn]),0,nx-1)
                                + nx*Numerics.bounded(Numerics.round(map2target[idn+ntxyz]),0,ny-1)
                                + nx*ny*Numerics.bounded(Numerics.round(map2target[idn+2*ntxyz]),0,nz-1);
                        if (mask[ngb]) {
                            if (labels[idn]==0) {
                                for (int best=0;best<nbest;best++) {
                                    if (combinedLabels[best][idmap[ngb]]==obj) {
                                        if (certainty) {
                                            if (best==0) {
                                                heap.addValue(combinedProbas[0][idmap[ngb]]-combinedProbas[1][idmap[ngb]],idn,obj);
                                            } else {
                                                heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[0][idmap[ngb]],idn,obj);
                                            }
                                        } else {
                                            heap.addValue(combinedProbas[best][idmap[ngb]],idn,obj);
                                        }
                                        best=nbest;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        System.out.println("Average volumes: ");
        for (int obj=1;obj<nobj;obj++) System.out.println(obj+": "+objVolumeMean[obj]+" ("+objVolumeStdv[obj]+") ");
        System.out.println("\nOptimized volumes: ");
        for (int obj=1;obj<nobj;obj++) System.out.println(obj+": "+bestvol[obj]+" ("+bestproba[obj]+") ");
        // re-run one last time to get the segmentation
        heap.reset();
        for (int obj=0;obj<nobj;obj++) {
            vol[obj] = 0.0;
        }
        for(int id=0;id<ntxyz;id++) labels[id] = 0;
        for (int obj=1;obj<nobj;obj++) {
            heap.addValue(bestscore[obj],start[obj],(byte)obj);
        }
        while (heap.isNotEmpty()) {
            float score = heap.getFirst();
            int idx = heap.getFirstId();
            int xyz = Numerics.bounded(Numerics.round(map2target[idx]),0,nx-1)
                    + nx*Numerics.bounded(Numerics.round(map2target[idx+ntxyz]),0,ny-1)
                    + nx*ny*Numerics.bounded(Numerics.round(map2target[idx+2*ntxyz]),0,nz-1);
            byte obj = heap.getFirstState();
            heap.removeFirst();
            if (labels[idx]==0) {
                if (vol[obj]<bestvol[obj]) {
                    // update the values
                    vol[obj] += rtx*rty*rtz;
                    labels[idx] = obj;
                
                    // add neighbors
                    for (byte k = 0; k<6; k++) {
                        int idn = Ngb.neighborIndex(k, idx, ntx, nty, ntz);
                        int ngb = Numerics.bounded(Numerics.round(map2target[idn]),0,nx-1)
                                + nx*Numerics.bounded(Numerics.round(map2target[idn+ntxyz]),0,ny-1)
                                + nx*ny*Numerics.bounded(Numerics.round(map2target[idn+2*ntxyz]),0,nz-1);
                        if (mask[ngb]) {
                            if (labels[idn]==0) {
                                for (int best=0;best<nbest;best++) {
                                    if (combinedLabels[best][idmap[ngb]]==obj) {
                                        if (certainty) {
                                            if (best==0) {
                                                heap.addValue(combinedProbas[0][idmap[ngb]]-combinedProbas[1][idmap[ngb]],idn,obj);
                                            } else {
                                                heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[0][idmap[ngb]],idn,obj);
                                            }
                                        } else {
                                            heap.addValue(combinedProbas[best][idmap[ngb]],idn,obj);
                                        }
                                        best=nbest;
                                    }
                                }
                            }  
                        }
                    }
                }
            }
        }
        // final segmentation: collapse on a single dimension
        finalLabel = labels;
        finalProba = new float[ntxyz];
        
        float[] kept = new float[nobj];
        for (int x=1;x<ntx-1;x++) for (int y=1;y<nty-1;y++) for (int z=1;z<ntz-1;z++) {
            int idx = x+ntx*y+ntx*nty*z;
            int xyz = Numerics.bounded(Numerics.round(map2target[idx]),0,nx-1)
                    + nx*Numerics.bounded(Numerics.round(map2target[idx+ntxyz]),0,ny-1)
                    + nx*ny*Numerics.bounded(Numerics.round(map2target[idx+2*ntxyz]),0,nz-1);
            if (mask[xyz]) {
                int id = idmap[xyz];
                int obj = labels[idx];
                for (int best=0;best<nbest;best++) {
                    if (combinedLabels[best][id]==obj) {
                        if (certainty) {
                            if (best==0) {
                                finalProba[idx] = combinedProbas[0][id]-combinedProbas[1][id];
                            } else {
                                finalProba[idx] = combinedProbas[best][id]-combinedProbas[0][id];
                            }
                        } else {
                            finalProba[idx] = combinedProbas[best][id];
                        }
                        best=nbest;
                    }
                }
            }
        }
        return;            
	}
}

