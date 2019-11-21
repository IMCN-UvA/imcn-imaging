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
	private float[][] mapImages = null;
	
	private float[][] targetImages;
	
	private double[][][] condmean = null;
	private double[][][] condstdv = null;
	
	private boolean[] mask = null;;
	
	private int nsub;
	private int nobj;
	private int nc;
	private int nbest = 16;
	
	private float deltaIn = 2.0f; // this parameter might have a big effect
	private float deltaOut = 0.0f;
	private float boundary = 10.0f; // tricky: it should be large enough not to crop enlarged structures
	// these have been tested: could be hard-coded
	private boolean modelBackground = true;
	private boolean cancelBackground = false;
	private boolean cancelAll = false;
	private boolean sumPosterior = false;
	private boolean maxPosterior = true;
	private int maxiter = 100;
	private float maxdiff = 0.1f;
	//private boolean topoParam = true;
	//private     String	            lutdir = null;
	// these have been tested: could be hard-coded
	private double top = 95.0;
	private boolean rescaleProbas = true;
	private boolean rescaleIntensities = true;
	private boolean modelHistogram = true;
	private boolean rescaleHistograms = true;
	
	// more things to tune? small & variable structures seem to vanish a bit fast
	private boolean scalePriors = true;
	
	private final float INF = 1e9f;
	
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
	
	private float[]        logVolMean;
	private float[]        logVolStdv;
	
	private float[][]        logVolMean2;
	private float[][]        logVolStdv2;
	
	private int nx, ny, nz, nxyz;
	private float rx, ry, rz;
	private int ndata;

	// in case the atlas and target spaces are different
	private float[]    map2atlas = null;
	private float[]    map2target = null;
	
	private int nax, nay, naz, naxyz;
	private float rax, ray, raz;

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
	public final void setMappingImageAt(int sub, float[] val) { 
	    if (mapImages==null) mapImages = new float[nsub][];
	    mapImages[sub] = val; 
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
	    System.out.println("atlas size: "+nax+" x "+nay+" x "+naz+" ("+naxyz+")");
	    
	    if (map2target!=null) {
	        System.out.println("image size: "+ntx+" x "+nty+" x "+ntz+" ("+ntxyz+")");
            mask = new boolean[ntxyz]; 
            for (int xt=0;xt<ntx;xt++) for (int yt=0;yt<nty;yt++) for (int zt=0;zt<ntz;zt++) {
                int idx = xt + ntx*yt + ntx*nty*zt;
                int xa = Numerics.bounded(Numerics.round(map2target[idx+0*ntxyz]),0,nax-1);
                int ya = Numerics.bounded(Numerics.round(map2target[idx+1*ntxyz]),0,nay-1);
                int za = Numerics.bounded(Numerics.round(map2target[idx+2*ntxyz]),0,naz-1);
                //int xyz =       Numerics.bounded(Numerics.round(map2target[idx+0*ntxyz]),0,nx-1)
                //        +    nx*Numerics.bounded(Numerics.round(map2target[idx+1*ntxyz]),0,ny-1)
                //        + nx*ny*Numerics.bounded(Numerics.round(map2target[idx+2*ntxyz]),0,nz-1);
                int xyz = xa + nax*ya + nax*nay*za;
                if (lval[xyz]>0) {
                    ndata++;
                    mask[idx] = true;
                } else {
                    mask[idx] = false;
                }
            }
            System.out.println("work region size: "+ndata);
            // build ID map
            idmap = new int[ntxyz];
            int id = 0;
            for (int idx=0;idx<ntxyz;idx++) if (mask[idx]) {
                idmap[idx] = id;
                id++;
            }
            System.out.println("id map size: "+id);
            // pass the probabilities USING the idmap
            spatialProbas = new float[nbest][ndata];
            spatialLabels = new int[nbest][ndata];
            for (int xt=0;xt<ntx;xt++) for (int yt=0;yt<nty;yt++) for (int zt=0;zt<ntz;zt++) {
                int idx = xt + ntx*yt + ntx*nty*zt;
                if (mask[idx]) {
                    int xa = Numerics.bounded(Numerics.round(map2target[idx+0*ntxyz]),0,nax-1);
                    int ya = Numerics.bounded(Numerics.round(map2target[idx+1*ntxyz]),0,nay-1);
                    int za = Numerics.bounded(Numerics.round(map2target[idx+2*ntxyz]),0,naz-1);
                    //int xyz =       Numerics.bounded(Numerics.round(map2target[idx+0*ntxyz]),0,nx-1)
                    //        +    nx*Numerics.bounded(Numerics.round(map2target[idx+1*ntxyz]),0,ny-1)
                    //        + nx*ny*Numerics.bounded(Numerics.round(map2target[idx+2*ntxyz]),0,nz-1);
                    int xyz = xa + nax*ya + nax*nay*za;

                    for (int best=0;best<nbest;best++) {
                        spatialProbas[best][idmap[idx]] = pval[xyz+best*naxyz];
                        spatialLabels[best][idmap[idx]] = lval[xyz+best*naxyz];
                    }
                }
            }
            // compute the structure-specific volume scaling factors
            // problem: if volume of average is null (low overlap + small structures), result is likely incorrect
            /*float[] atlasVol = new float[nobj];
            for (int xyz=0;xyz<naxyz;xyz++) if (lval[xyz]>0) {
                int obj = Numerics.floor(lval[xyz]/100.0f);
                atlasVol[obj-1] += rax*ray*raz;
            }
            float[] mappedVol = new float[nobj];
            for (int idx=0;idx<ntxyz;idx++) if (mask[idx]) {
                int obj = Numerics.floor(spatialLabels[0][idmap[idx]]/100.0f);
                mappedVol[obj-1] += rtx*rty*rtz;
            }*/
            // use a region growing approach instead...
            int[] atlasLabels = atlasVolumeLabels(pval, lval);
            float[] atlasVol = new float[nobj];
            for (int xyz=0;xyz<naxyz;xyz++) if (atlasLabels[xyz]>0) {
                atlasVol[atlasLabels[xyz]] += rax*ray*raz;
            }
            System.out.print("atlas volumes: ");
            for (int obj=0;obj<nobj;obj++) System.out.print(obj+": "+(atlasVol[obj])+", ");
            System.out.println("(a)");
            
            float[] mappedVol = new float[nobj];
            for (int xt=0;xt<ntx;xt++) for (int yt=0;yt<nty;yt++) for (int zt=0;zt<ntz;zt++) {
                int idx = xt + ntx*yt + ntx*nty*zt;
                if (mask[idx]) {
                    int xa = Numerics.bounded(Numerics.round(map2target[idx+0*ntxyz]),0,nax-1);
                    int ya = Numerics.bounded(Numerics.round(map2target[idx+1*ntxyz]),0,nay-1);
                    int za = Numerics.bounded(Numerics.round(map2target[idx+2*ntxyz]),0,naz-1);
                    //int xyz =       Numerics.bounded(Numerics.round(map2target[idx+0*ntxyz]),0,nx-1)
                    //        +    nx*Numerics.bounded(Numerics.round(map2target[idx+1*ntxyz]),0,ny-1)
                    //        + nx*ny*Numerics.bounded(Numerics.round(map2target[idx+2*ntxyz]),0,nz-1);
                    int xyz = xa + nax*ya + nax*nay*za;

                    if (atlasLabels[xyz]>0) {
                        mappedVol[atlasLabels[xyz]] += rtx*rty*rtz;
                    }
                }
            }
            System.out.print("mapped volumes: ");
            for (int obj=0;obj<nobj;obj++) System.out.print(obj+": "+(mappedVol[obj])+", ");
            System.out.println("(m)");
            
            System.out.print("volume ratios: ");
            for (int obj=0;obj<nobj;obj++) System.out.print(obj+": "+(mappedVol[obj]/atlasVol[obj])+", ");
            System.out.println("(m/a)");
            for (int obj=0;obj<nobj;obj++) {
                //objVolumeMean[obj] *= mappedVol[obj]/atlasVol[obj];
                //objVolumeStdv[obj] *= mappedVol[obj]/atlasVol[obj];
                logVolMean[obj] += FastMath.log(mappedVol[obj]) - FastMath.log(atlasVol[obj]);
                logVolStdv[obj] += FastMath.log(mappedVol[obj]) - FastMath.log(atlasVol[obj]);
            }
            
            // adjust the volume boudnary priors too...
            int[] atlasBoundaryLabels = atlasBoundaryLabels(pval, lval);
            float[][] atlasVol2 = new float[nobj][nobj];
            for (int xyz=0;xyz<naxyz;xyz++) if (atlasBoundaryLabels[xyz]>0) {
                int obj1 = Numerics.floor(atlasBoundaryLabels[xyz]/100)-1;
                int obj2 = atlasBoundaryLabels[xyz] - 100*(obj1+1) - 1;
                
                atlasVol2[obj1][obj2] += rax*ray*raz;
            }
            System.out.print("atlas boudnary volumes: ");
            for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) 
                System.out.print(obj1+" | "+obj2+" : "+(atlasVol2[obj1][obj2])+", ");
            System.out.println("(a)");
            
            float[][] mappedVol2 = new float[nobj][nobj];
            for (int xt=0;xt<ntx;xt++) for (int yt=0;yt<nty;yt++) for (int zt=0;zt<ntz;zt++) {
                int idx = xt + ntx*yt + ntx*nty*zt;
                if (mask[idx]) {
                    int xa = Numerics.bounded(Numerics.round(map2target[idx+0*ntxyz]),0,nax-1);
                    int ya = Numerics.bounded(Numerics.round(map2target[idx+1*ntxyz]),0,nay-1);
                    int za = Numerics.bounded(Numerics.round(map2target[idx+2*ntxyz]),0,naz-1);
                    //int xyz =       Numerics.bounded(Numerics.round(map2target[idx+0*ntxyz]),0,nx-1)
                    //        +    nx*Numerics.bounded(Numerics.round(map2target[idx+1*ntxyz]),0,ny-1)
                    //        + nx*ny*Numerics.bounded(Numerics.round(map2target[idx+2*ntxyz]),0,nz-1);
                    int xyz = xa + nax*ya + nax*nay*za;

                    if (atlasBoundaryLabels[xyz]>0) {
                        int obj1 = Numerics.floor(atlasBoundaryLabels[xyz]/100)-1;
                        int obj2 = atlasBoundaryLabels[xyz] - 100*(obj1+1) - 1;
                        
                        mappedVol2[obj1][obj2] += rtx*rty*rtz;
                    }
                }
            }
            System.out.print("mapped volumes: ");
            for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) 
                System.out.print(obj1+" | "+obj2+" : "+(mappedVol2[obj1][obj2])+", ");
            System.out.println("(m)");
            
            System.out.print("volume ratios: ");
            for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                if (mappedVol2[obj1][obj2]>0 && atlasVol2[obj1][obj2]>0) {
                    System.out.print(obj1+" | "+obj2+" : "+(mappedVol2[obj1][obj2]/atlasVol2[obj1][obj2])+", ");
                } else {
                    System.out.print(obj1+" | "+obj2+" : n/a, ");
                }
            }
            System.out.println("(m/a)");
            // adjust the volume priors
            for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                //objVolumeMean[obj] *= mappedVol[obj]/atlasVol[obj];
                //objVolumeStdv[obj] *= mappedVol[obj]/atlasVol[obj];
                logVolMean2[obj1][obj2] += FastMath.log(Numerics.max(1.0,mappedVol2[obj1][obj2])) - FastMath.log(Numerics.max(1.0,atlasVol2[obj1][obj2]));
                logVolStdv2[obj1][obj2] += FastMath.log(Numerics.max(1.0,mappedVol2[obj1][obj2])) - FastMath.log(Numerics.max(1.0,atlasVol2[obj1][obj2]));
            }
            // reset all indices to subject space
            nx = ntx; ny = nty; nz = ntz; nxyz = ntxyz;
            rx = rtx; ry = rty; rz = rtz;
            map2target = null;
            map2atlas = null;
	    } else {
            mask = new boolean[naxyz]; 
            for (int xyz=0;xyz<naxyz;xyz++) {
                if (lval[xyz]>0) {
                    mask[xyz] = true;
                    ndata++;
                } else {
                    mask[xyz] = false;
                }
            }
            System.out.println("work region size: "+ndata);
            // build ID map
            idmap = new int[naxyz];
            int id = 0;
            for (int xyz=0;xyz<naxyz;xyz++) if (mask[xyz]) {
                idmap[xyz] = id;
                id++;
            }
            // pass the probabilities USING the idmap
            spatialProbas = new float[nbest][ndata];
            spatialLabels = new int[nbest][ndata];
            for (int xyz=0;xyz<naxyz;xyz++) if (mask[xyz]) {
                for (int best=0;best<nbest;best++) {
                    spatialProbas[best][idmap[xyz]] = pval[xyz+best*naxyz];
                    spatialLabels[best][idmap[xyz]] = lval[xyz+best*naxyz];
                }
            }
            nx = nax; ny = nay; nz = naz; nxyz = naxyz;
            rx = rax; ry = ray; rz = raz;
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
		logVolMean = new float[nobj];
		logVolStdv = new float[nobj];
	    logVolMean2 = new float[nobj][nobj];
		logVolStdv2 = new float[nobj][nobj];
	    for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) for (int c=0;c<nc;c++) {
	        condpair[c][obj1][obj2] = false;
	        condmin[c][obj1][obj2] = val[obj2+obj1*nobj+nobj*nobj*0+nobj*nobj*(nbins+6)*c];
	        for (int bin=0;bin<nbins;bin++) {
	            condhistogram[c][obj1][obj2][bin] = val[obj2+obj1*nobj+nobj*nobj*(bin+1)+nobj*nobj*(nbins+6)*c];
	            if (condhistogram[c][obj1][obj2][bin]>0) condpair[c][obj1][obj2] = true;
	        }
	        condmax[c][obj1][obj2] = val[obj2+obj1*nobj+nobj*nobj*(nbins+1)+nobj*nobj*(nbins+6)*c];
	        if (obj1==obj2) {
	            logVolMean[obj1] = val[obj2+obj1*nobj+nobj*nobj*(nbins+2)+nobj*nobj*(nbins+6)*c];
	            logVolStdv[obj1] = val[obj2+obj1*nobj+nobj*nobj*(nbins+3)+nobj*nobj*(nbins+6)*c];
	        }
            logVolMean2[obj1][obj2] = val[obj2+obj1*nobj+nobj*nobj*(nbins+4)+nobj*nobj*(nbins+6)*c];
            logVolStdv2[obj1][obj2] = val[obj2+obj1*nobj+nobj*nobj*(nbins+5)+nobj*nobj*(nbins+6)*c];
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

	public final void setAtlasDimensions(int x, int y, int z) { nax=x; nay=y; naz=z; naxyz=nax*nay*naz; }
	public final void setAtlasDimensions(int[] dim) { nax=dim[0]; nay=dim[1]; naz=dim[2]; naxyz=nax*nay*naz; }
	
	public final void setAtlasResolutions(float x, float y, float z) { rax=x; ray=y; raz=z; }
	public final void setAtlasResolutions(float[] res) { rax=res[0]; ray=res[1]; raz=res[2]; }
	
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
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int best=0;best<nval;best++) {
                images[xyz+best*nxyz] = spatialProbas[best][idmap[xyz]];
            }
        }
        return images;
	}
	public final int[] getBestSpatialProbabilityLabels(int nval) {
        nval = Numerics.min(nval,nbest);
        int[] images = new int[nval*nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int best=0;best<nval;best++) {
                images[xyz+best*nxyz] = spatialLabels[best][idmap[xyz]];
            }
        }
        return images;
	}

	public final float[] getBestIntensityProbabilityMaps(int nval) {
        nval = Numerics.min(nval,nbest);
        float[] images = new float[nval*nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int best=0;best<nval;best++) {
                images[xyz+best*nxyz] = intensityProbas[best][idmap[xyz]];
            }
        }
        return images;
	}
	public final int[] getBestIntensityProbabilityLabels(int nval) {
        nval = Numerics.min(nval,nbest);
        int[] images = new int[nval*nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int best=0;best<nval;best++) {
                images[xyz+best*nxyz] = intensityLabels[best][idmap[xyz]];
            }
        }
        return images;
	}

    public final float[] getBestProbabilityMaps(int nval) { 
        nval = Numerics.min(nval,nbest);
        float[] images = new float[nval*nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int best=0;best<nval;best++) {
                images[xyz+best*nxyz] = combinedProbas[best][idmap[xyz]];
            }
        }
        return images;
	}
	public final int[] getBestProbabilityLabels(int nval) { 
        nval = Numerics.min(nval,nbest);
        int[] images = new int[nval*nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int best=0;best<nval;best++) {
                images[xyz+best*nxyz] = combinedLabels[best][idmap[xyz]];
            }
        }
        return images;
	}
	
	public final float[] getCertaintyProbability() { 
        float[] images = new float[nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    images[xyz] = combinedProbas[0][idmap[xyz]]-combinedProbas[1][idmap[xyz]];
        }
        return images;
	}
    public final float[] getNeighborhoodMaps(int nval) { 
        float[] images = new float[nval*nxyz];
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int best=0;best<nval;best++) {
                images[xyz+best*nxyz] = ngbw[best][idmap[xyz]];
            }
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
	    float[] val = new float[nc*nobj*nobj*(nbins+6)];
	    for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) for (int c=0;c<nc;c++) {
	        val[obj2+obj1*nobj+0*nobj*nobj+c*nobj*nobj*(nbins+6)] = (float)condmin[c][obj1][obj2];
	        for (int bin=0;bin<nbins;bin++) {
	            val[obj2+obj1*nobj+(bin+1)*nobj*nobj+c*nobj*nobj*(nbins+6)] = (float)condhistogram[c][obj1][obj2][bin];
	        }
	        val[obj2+obj1*nobj+(nbins+1)*nobj*nobj+c*nobj*nobj*(nbins+6)] = (float)condmax[c][obj1][obj2];
	        if (obj1==obj2) {
	            val[obj2+obj1*nobj+(nbins+2)*nobj*nobj+c*nobj*nobj*(nbins+6)] = (float)logVolMean[obj1];
	            val[obj2+obj1*nobj+(nbins+3)*nobj*nobj+c*nobj*nobj*(nbins+6)] = (float)logVolStdv[obj1];
	        }
            val[obj2+obj1*nobj+(nbins+4)*nobj*nobj+c*nobj*nobj*(nbins+6)] = (float)logVolMean2[obj1][obj2];
            val[obj2+obj1*nobj+(nbins+5)*nobj*nobj+c*nobj*nobj*(nbins+6)] = (float)logVolStdv2[obj1][obj2];
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
	    strictSimilarityDiffusion(4);
	    collapseConditionalMaps();
    }	    
	    
	public final void computeAtlasPriors() {
	    nx = nax; ny = nay; nz = naz; nxyz = naxyz;
	    rx = rax; ry = ray; rz = raz;
	    
	    float[][][] levelsets = null; 
	    
	    // not correct: explicitly build the levelset of the background first, then crop it
	    if (modelBackground) {
            // adding the background: building a ring around the structures of interest
            // with also a sharp decay to the boundary
            levelsets = new float[nsub][nobj][];
            float[] background = new float[nxyz];
            boolean[] bgmask = new boolean[nxyz];
            for (int xyz=0;xyz<nxyz;xyz++) bgmask[xyz] = true;
            boundary = boundary/Numerics.max(rx,ry,rz);
            
            for (int sub=0;sub<nsub;sub++) {
                for (int xyz=0;xyz<nxyz;xyz++) {
                    float mindist = boundary;
                    for (int obj=0;obj<nobj-1;obj++) {
                        if (lvlImages[sub][obj][xyz]<mindist) mindist = lvlImages[sub][obj][xyz];
                    }
                    /*
                    if (mindist<boundary) {
                        background[xyz] = -mindist;
                    } else {
                        background[xyz] = mindist;
                    }*/
                    background[xyz] = -mindist;
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
		    // skip the background label (bg must be the first label always)
            for (int sub=0;sub<nsub;sub++) for (int obj=1;obj<nobj;obj++) {
                if (levelsets[sub][obj][xyz]<mindist) mindist = levelsets[sub][obj][xyz];
            }
            if (mindist<boundary) {
                mask[xyz] = true;
                ndata++;
            } else {
                mask[xyz] = false;
            }
        }
        // build ID map
        idmap = new int[ntxyz];
        int id = 0;
        for (int xyz=0;xyz<ntxyz;xyz++) if (mask[xyz]) {
            idmap[xyz] = id;
            id++;
        }
        System.out.println("masking: work region "+ndata+", compression: "+(ndata/(float)nxyz));
		
        // adapt number of kept values?
        
		System.out.println("compute joint conditional shape priors");
		spatialProbas = new float[nbest][ndata]; 
		spatialLabels = new int[nbest][ndata];
		
		int ctr = Numerics.floor(nsub/2);
        int dev = Numerics.floor(nsub/4);
                    
		double[] val = new double[nsub];
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
                // when scaling by the variance, it penalizes more strongly variable regions -> they get a weaker prior
                // maybe a good thing? not entirely sure...
                if (scalePriors)
                    priors[obj1][obj2] = 1.0/FastMath.sqrt(2.0*FastMath.PI*sigma2)*FastMath.exp( -0.5*mean*mean/sigma2 );
                else
                    priors[obj1][obj2] = FastMath.exp( -0.5*mean*mean/sigma2 );
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
                    spatialLabels[best][idmap[xyz]] = 100*(best1+1)+(best2+1);
                    spatialProbas[best][idmap[xyz]] = (float)priors[best1][best2];
                } else {
                    for (int b=best;b<nbest;b++) {
                        spatialLabels[b][idmap[xyz]] = 0;
                        spatialProbas[b][idmap[xyz]] = 0.0f;
                    }
                    best = nbest;
                }                    
                // remove best value
                priors[best1][best2] = 0.0;
 		    }
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
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int c=0;c<nc;c++) {
                for  (int sub=0;sub<nsub;sub++) {
                    val[sub] = contrasts[sub][c][xyz];
                }
                /*
                //System.out.println("values");
                Percentile measure = new Percentile();
                measure.setData(val);
			
                medc[c][idmap[xyz]] = (float)measure.evaluate(50.0); 
                //System.out.println("median "+medc[c][idmap[xyz]]);
                iqrc[c][idmap[xyz]] = (float)(measure.evaluate(75.0) - measure.evaluate(25.0));
                //System.out.println("iqr "+iqrc[c][idmap[xyz]]);
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
                medc[c][idmap[xyz]] = (float)med;
                iqrc[c][idmap[xyz]] = (float)iqr;
                
                cntsum[c] += iqr;
                cntden[c]++;
            }
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
                    for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
                        double med = medc[c][idmap[xyz]];
                        double iqr = iqrc[c][idmap[xyz]];
                        // assuming here that iqr==0 means masked regions
                        if (iqr>0) { 
                            // look only among non-zero priors for each region
                            for (int best=0;best<nbest;best++) {
                                if (spatialLabels[best][idmap[xyz]]==100*(obj1+1)+(obj2+1)) {
                                    // found value: proceeed
                                    if (med<condmin[c][obj1][obj2]) condmin[c][obj1][obj2] = med;
                                    if (med>condmax[c][obj1][obj2]) condmax[c][obj1][obj2] = med;
                                    if (condmin[c][obj1][obj2]!=condmax[c][obj1][obj2]) existsPair = true;
                                }
                            }
                        }
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
                        for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
                            double med = medc[c][idmap[xyz]];
                            double iqr = iqrc[c][idmap[xyz]];
                            // assuming here that iqr==0 means masked regions
                            if (iqr>0) { 
                                // look for non-zero priors
                                for (int best=0;best<nbest;best++) {
                                    if (spatialLabels[best][idmap[xyz]]==100*(obj1+1)+(obj2+1)) {
                                        // found value: proceeed
                                        for (int sub=0;sub<nsub;sub++) {
                                            // adds uncertainties from mismatch between subject intensities and mean shape
                                            /*
                                            double psub = spatialProbas[best][idmap[xyz]]*1.0/FastMath.sqrt(2.0*FastMath.PI*1.349*iqr*1.349*iqr)
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
                   for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
                       double med = medc[c][idmap[xyz]];
                       double iqr = iqrc[c][idmap[xyz]];
                       // assuming here that iqr==0 means masked regions
                       if (iqr>0) { 
                           // look for non-zero priors
                           for (int best=0;best<nbest;best++) {
                               if (spatialLabels[best][idmap[xyz]]==100*(obj1+1)+(obj2+1)) {
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
                   for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
                       double med = medc[c][idmap[xyz]];
                       double iqr = iqrc[c][idmap[xyz]];
                       // assuming here that iqr==0 means masked regions
                       if (iqr>0) { 
                           // look for non-zero priors
                           for (int best=0;best<nbest;best++) {
                               if (spatialLabels[best][idmap[xyz]]==100*(obj1+1)+(obj2+1)) {
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
        System.out.println("Volume statistics");
        logVolMean = new float[nobj];
        logVolStdv = new float[nobj];
        for (int obj=0;obj<nobj;obj++) {
            float[] vols = new float[nsub];
            for (int sub=0;sub<nsub;sub++) {
                vols[sub] = 0.0f;
                for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
                    if (levelsets[sub][obj][xyz]<0) {
                        vols[sub]+=rx*ry*rz;
                    }
                }
                logVolMean[obj] += FastMath.log(Numerics.max(1.0,vols[sub]))/nsub;
            }
            for (int sub=0;sub<nsub;sub++) {
                logVolStdv[obj] += Numerics.square(FastMath.log(Numerics.max(1.0,vols[sub]))-logVolMean[obj])/(nsub-1.0f);
            }
            logVolStdv[obj] = (float)FastMath.sqrt(logVolStdv[obj]);
            System.out.println(obj+" : "+FastMath.exp(logVolMean[obj])
                                   +" ["+FastMath.exp(logVolMean[obj]-logVolStdv[obj])
                                   +", "+FastMath.exp(logVolMean[obj]+logVolStdv[obj]));
        }        
        // same for all interfaces
        System.out.println("Conditional volume statistics");
        logVolMean2 = new float[nobj][nobj];
        logVolStdv2 = new float[nobj][nobj];
        float[][][] vols = new float[nsub][nobj][nobj];
        for (int sub=0;sub<nsub;sub++) {
            for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
                double best = INF;
                int best1 = -1;
                int best2 = -1;
                for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                    if (Numerics.max(0.0, levelsets[sub][obj1][xyz]-deltaOut, levelsets[sub][obj2][xyz]-deltaIn)<best) {
                        best = Numerics.max(0.0, levelsets[sub][obj1][xyz]-deltaOut, levelsets[sub][obj2][xyz]-deltaIn);
                        best1 = obj1;
                        best2 = obj2;
                    }
                }
                vols[sub][best1][best2]+=rx*ry*rz;
            }
            for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                logVolMean2[obj1][obj2] += FastMath.log(Numerics.max(1.0,vols[sub][obj1][obj2]))/nsub;
            }
        }
        for (int sub=0;sub<nsub;sub++) {
            for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                logVolStdv2[obj1][obj2] += Numerics.square(FastMath.log(Numerics.max(1.0,vols[sub][obj1][obj2]))-logVolMean2[obj1][obj2])/(nsub-1.0f);
            }
        }
        for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
            logVolStdv2[obj1][obj2] = (float)FastMath.sqrt(logVolStdv2[obj1][obj2]);
            System.out.println(obj1+"|"+obj2+" : "+FastMath.exp(logVolMean2[obj1][obj2])
                                             +" ["+FastMath.exp(logVolMean2[obj1][obj2]-logVolStdv2[obj1][obj2])
                                             +", "+FastMath.exp(logVolMean2[obj1][obj2]+logVolStdv2[obj1][obj2])+"]");
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
                           if (spatialLabels[best][idmap[xyz]]==100*(obj1+1)+(obj2+1)) {
                               // multiply nc times to balance prior and posterior
                               //likelihood[obj1][obj2] = 1.0;
                               likelihood[obj1][obj2] = spatialProbas[best][idmap[xyz]];
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
                       if (spatialLabels[sbest][idmap[xyz]]==100*(best1+1)+(best2+1)) {
                           shapeprior = spatialProbas[sbest][idmap[xyz]];
                           sbest = nbest;
                       }
                    }
                    // sub optimal labeling, but easy to read
                    separateIntensLabels[c][best][idmap[xyz]] = 100*(best1+1)+(best2+1);
                    separateIntensProbas[c][best][idmap[xyz]] = (float)(likelihood[best1][best2]/shapeprior);
                    // remove best value
                    likelihood[best1][best2] = 0.0;
                }
             }
            if (rescaleIntensities) {
                // rescale top % in each shape and intensity priors
                Percentile measure = new Percentile();
                double[] val = new double[ndata];
                for (int id=0;id<ndata;id++) val[id] = separateIntensProbas[c][0][id];
                float intensMax = (float)measure.evaluate(val, top);
                System.out.println("top "+top+"% intensity probability (contrast "+c+"): "+intensMax);
                
                for (int id=0;id<ndata;id++) for (int best=0;best<nbest;best++) {
                    separateIntensProbas[c][best][id] = (float)Numerics.min(top/100.0*separateIntensProbas[c][best][id]/intensMax, 1.0f);
                }
            }
		}
        // combine the multiple contrasts            
		System.out.println("combine intensity probabilities");
		intensityProbas = new float[nbest][ndata]; 
		intensityLabels = new int[nbest][ndata];
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
            double[][] likelihood = new double[nobj][nobj];
            for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
               likelihood[obj1][obj2] = 1.0;
               for (int c=0;c<nc;c++) {
                   double val = 0.0;
                   for (int best=0;best<nbest;best++) {
                       if (separateIntensLabels[c][best][idmap[xyz]]==100*(obj1+1)+(obj2+1)) {
                           val = separateIntensProbas[c][best][idmap[xyz]];
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
                intensityLabels[best][idmap[xyz]] = 100*(best1+1)+(best2+1);
                // scaling for multiplicative intensities
                intensityProbas[best][idmap[xyz]] = (float)FastMath.pow(likelihood[best1][best2],1.0/nc);
                // remove best value
                likelihood[best1][best2] = 0.0;
            }
        }
		if (!rescaleIntensities && rescaleProbas) {
            Percentile measure = new Percentile();
            double[] val = new double[ndata];
            for (int id=0;id<ndata;id++) val[id] = intensityProbas[0][id];
            float intensMax = (float)measure.evaluate(val, top);
            System.out.println("top "+top+"% global intensity probability: "+intensMax);
            for (int id=0;id<ndata;id++) for (int best=0;best<nbest;best++) {
                intensityProbas[best][id] = (float)Numerics.min(top/100.0*intensityProbas[best][id]/intensMax, 1.0f);
            }		
		}
		
		// posterior : merge both measures
		System.out.println("generate posteriors");
        combinedProbas = new float[nbest][ndata]; 
		combinedLabels = new int[nbest][ndata];
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    double[][] posteriors = new double[nobj][nobj];
		    for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                // look for non-zero priors
                posteriors[obj1][obj2] = 0.0;
                
                for (int best=0;best<nbest;best++) {
                    if (spatialLabels[best][idmap[xyz]]==100*(obj1+1)+(obj2+1)) {
                        // multiply nc times to balance prior and posterior
                        posteriors[obj1][obj2] = spatialProbas[best][idmap[xyz]];
                        best = nbest;
                    }
                }
                if (posteriors[obj1][obj2]>0) {
                    double intensPrior = 0.0;
                    for (int best=0;best<nbest;best++) {
                        if (intensityLabels[best][idmap[xyz]]==100*(obj1+1)+(obj2+1)) {
                            intensPrior = intensityProbas[best][idmap[xyz]];
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
                combinedLabels[best][idmap[xyz]] = 100*(best1+1)+(best2+1);
                combinedProbas[best][idmap[xyz]] = (float)FastMath.sqrt(posteriors[best1][best2]);
                // remove best value
                posteriors[best1][best2] = 0.0;
 		    }
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
                //combinedLabels[best][idmap[xyz]] = 100*(best1+1)+(best2+1);
                combinedLabels[best][id] = best1;
                combinedProbas[best][id] = (float)posteriors[best1][best2];
                // remove best value
                posteriors[best1][best2] = 0.0;
 		    }
        }
    }

	public final void collapseSpatialPriorMaps() {	    
        for (int id=0;id<ndata;id++) {
            double[][] priors = new double[nobj][nobj];
            for (int best=0;best<nbest;best++) {
                int obj1 = Numerics.floor(spatialLabels[best][id]/100)-1;
                int obj2 = spatialLabels[best][id]-(obj1+1)*100-1;
                priors[obj1][obj2] = spatialProbas[best][id];
            }
            if (sumPosterior) {
                for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) if (obj2!=obj1) {
                   priors[obj1][obj1] += priors[obj1][obj2];
                   priors[obj1][obj2] = 0.0;
                }
            } else if (maxPosterior) {
                for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) if (obj2!=obj1) {
                    priors[obj1][obj1] = Numerics.max(priors[obj1][obj1],priors[obj1][obj2]);
                    priors[obj1][obj2] = 0.0;
                }
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
                // sub optimal labeling, but easy to read
                //combinedLabels[best][idmap[xyz]] = 100*(best1+1)+(best2+1);
                spatialLabels[best][id] = best1;
                spatialProbas[best][id] = (float)priors[best1][best2];
                // remove best value
                priors[best1][best2] = 0.0;
 		    }
        }
    }

	public final void strictSimilarityDiffusion(int nngb) {	
		
		float[][] target = targetImages;
		// add a local diffusion step?
		System.out.print("Diffusion step: \n");
		
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
		for (int id=0;id<ndata;id++) {
		    for (int best=0;best<nbest;best++) {
                diffusedProbas[best][id] = combinedProbas[best][id];
                diffusedLabels[best][id] = combinedLabels[best][id];
            }
        }
		    		
		double[][] diffused = new double[nobj][nobj];
        for (int t=0;t<maxiter;t++) {
		    for (int id=0;id<ndata;id++) if (ngbw[nngb][id]>0) {
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
            for (int id=0;id<ndata;id++) for (int best=0;best<nbest;best++) {
                if (combinedLabels[best][id] == diffusedLabels[best][id]) {
                    //diff += Numerics.abs(diffusedProbas[best][idmap[xyz]]-combinedProbas[best][idmap[xyz]]);
                    diff += 0.0;
                } else {
                    //diff += Numerics.abs(diffusedProbas[best][idmap[xyz]]-combinedProbas[best][idmap[xyz]]);
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
	
	public final void fastSimilarityDiffusion(int nngb) {	
		
		float[][] target = targetImages;
		// add a local diffusion step?
		System.out.print("Diffusion step: \n");
		
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
		for (int id=0;id<ndata;id++) {
		    for (int best=0;best<nbest;best++) {
                diffusedProbas[best][id] = combinedProbas[best][id];
                diffusedLabels[best][id] = combinedLabels[best][id];
            }
        }
		    		
		double[][] diffused = new double[nobj][nobj];
        for (int t=0;t<maxiter;t++) {
		    for (int id=0;id<ndata;id++) if (ngbw[nngb][id]>0) {
                for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                    diffused[obj1][obj2] = 0.0;
                }
                
                for (int best=0;best<nbest;best++) {
                    int obj1 = Numerics.floor(combinedLabels[best][id]/100.0f)-1;
                    int obj2 = Numerics.round(combinedLabels[best][id]-100*(obj1+1)-1);
                     
                    diffused[obj1][obj2] = combinedProbas[best][id];
                    float den = ngbw[nngb][id];
                    diffused[obj1][obj2] *= den;
                    
                    for (int n=0;n<nngb;n++) {
                        int ngb = ngbi[n][id];
                        float ngbmax = 0.0f;
                        // max over neighbors ( -> stop at first found)
                        for (int bestngb=0;bestngb<nbest;bestngb++) {
                            if (combinedLabels[bestngb][ngb]==combinedProbas[best][id]) {
                                ngbmax = combinedProbas[bestngb][ngb];
                                bestngb=nbest;
                            }
                        }
                        //if (ngbmax==0) System.out.print("0");
                        diffused[obj1][obj2] += ngbw[n][id]*ngbmax;
                        den += ngbw[n][id];
                    }
                    diffused[obj1][obj2] /= den;
                }
                for (int best=0;best<nbest;best++) {
                    int best1 = Numerics.floor(combinedLabels[best][id]/100.0f)-1;
                    int best2 = Numerics.round(combinedLabels[best][id]-100*(best1+1)-1);
                        
                    for (int next=0;next<nbest;next++) {
                        int obj1 = Numerics.floor(combinedLabels[next][id]/100.0f)-1;
                        int obj2 = Numerics.round(combinedLabels[next][id]-100*(obj1+1)-1);
                        if (diffused[obj1][obj2]>diffused[best1][best2]) {
                            best1 = obj1;
                            best2 = obj2;
                        }
                        // sub optimal labeling, but easy to read
                        diffusedLabels[best][id] = 100*(best1+1)+(best2+1);
                        diffusedProbas[best][id] = (float)diffused[best1][best2];
                        // remove best value
                        diffused[best1][best2] = 0.0;
                    }
                }
            }
            double diff = 0.0;
            for (int id=0;id<ndata;id++) for (int best=0;best<nbest;best++) {
                if (combinedLabels[best][id] == diffusedLabels[best][id]) {
                    //diff += Numerics.abs(diffusedProbas[best][idmap[xyz]]-combinedProbas[best][idmap[xyz]]);
                    diff += 0.0;
                } else {
                    //diff += Numerics.abs(diffusedProbas[best][idmap[xyz]]-combinedProbas[best][idmap[xyz]]);
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
		
	private int[] atlasVolumeLabels(float[] pval, int[] lval) {
		// find appropriate threshold to have correct volume; should use a fast marching approach!
		BinaryHeap2D	heap = new BinaryHeap2D(nax*nay+nay*naz+naz*nax, BinaryHeap4D.MAXTREE);
		int[] labels = new int[naxyz];
        int[] start = new int[nobj];
        float[] bestscore = new float[nobj];
        for (int obj=1;obj<nobj;obj++) bestscore[obj] = -INF;
        heap.reset();
		// important: skip first label as background (allows for unbounded growth)
        for (int obj=1;obj<nobj;obj++) {
		    // find highest scoring voxel as starting point
            for (int b=0;b<nbest-1;b++) {
                for (int x=1;x<nax-1;x++) for (int y=1;y<nay-1;y++) for (int z=1;z<naz-1;z++) {
                    int xyz=x+nax*y+nax*nay*z;
                    if (Numerics.floor(lval[xyz+b*naxyz]/100.0f)==obj+1) {
                        if (pval[xyz+b*naxyz]>bestscore[obj]) {
                            bestscore[obj] = pval[xyz+b*naxyz];
                            start[obj] = xyz;
                        }
                    }
                }
                if (bestscore[obj]>-INF) b = nbest;
            }
            heap.addValue(bestscore[obj],start[obj],(byte)obj);
        }
        double[] vol = new double[nobj];
        double[] volmean = new double[nobj];
        for (int obj=1;obj<nobj;obj++) volmean[obj] = FastMath.exp(logVolMean[obj]);
        while (heap.isNotEmpty()) {
            float score = heap.getFirst();
            int xyz = heap.getFirstId();
            byte obj = heap.getFirstState();
            heap.removeFirst();
            if (labels[xyz]==0) {
                 if (vol[obj]<volmean[obj]) {
                    // update the values
                    vol[obj]+=rax*ray*raz;
                    labels[xyz] = obj;
                
                    // add neighbors
                    for (byte k = 0; k<26; k++) {
                        int ngb = Ngb.neighborIndex(k, xyz, nax, nay, naz);
                        if (ngb>0 && ngb<naxyz) {
                           if (labels[ngb]==0) {
                                for (int best=0;best<nbest;best++) {
                                    if (Numerics.floor(lval[xyz+best*naxyz]/100.0f)==obj+1) {
                                       heap.addValue(pval[ngb+best*naxyz],ngb,obj);
                                       best=nbest;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return labels;            
	}
			
	private int[] atlasBoundaryLabels(float[] pval, int[] lval) {
		// find appropriate threshold to have correct volume; should use a fast marching approach!
		BinaryHeapPair	heap = new BinaryHeapPair(nax*nay+nay*naz+naz*nax, BinaryHeapPair.MAXTREE);
		int[] labels = new int[naxyz];
        int[] start = new int[nobj];
        float[] bestscore = new float[nobj];
        for (int obj=1;obj<nobj;obj++) bestscore[obj] = -INF;
        heap.reset();
		// important: skip first label as background (allows for unbounded growth)
        for (int obj=1;obj<nobj;obj++) {
		    // find highest scoring voxel as starting point
            for (int b=0;b<nbest-1;b++) {
                for (int x=1;x<nax-1;x++) for (int y=1;y<nay-1;y++) for (int z=1;z<naz-1;z++) {
                    int xyz=x+nax*y+nax*nay*z;
                    if (Numerics.floor(lval[xyz+b*naxyz]/100.0f)==obj+1) {
                        if (pval[xyz+b*naxyz]>bestscore[obj]) {
                            bestscore[obj] = pval[xyz+b*naxyz];
                            start[obj] = xyz;
                        }
                    }
                }
                if (bestscore[obj]>-INF) b = nbest;
            }
            heap.addValue(bestscore[obj],start[obj],101*(obj+1));
        }
        double[] vol = new double[nobj];
        double[] volmean = new double[nobj];
        for (int obj=1;obj<nobj;obj++) volmean[obj] = FastMath.exp(logVolMean[obj]);
        while (heap.isNotEmpty()) {
            float score = heap.getFirst();
            int xyz = heap.getFirstId1();
            int obj1obj2 = heap.getFirstId2();
            int obj1 = Numerics.floor(obj1obj2/100.0f)-1;
            int obj2 = Numerics.round(obj1obj2-100*(obj1+1)-1);
            heap.removeFirst();
            if (labels[xyz]==0) {
                 if (vol[obj1]<volmean[obj1]) {
                    // update the values
                    vol[obj1]+=rax*ray*raz;
                    labels[xyz] = obj1obj2;
                
                    // add neighbors
                    for (byte k = 0; k<26; k++) {
                        int ngb = Ngb.neighborIndex(k, xyz, nax, nay, naz);
                        if (ngb>0 && ngb<naxyz) {
                           if (labels[ngb]==0) {
                                for (int best=0;best<nbest;best++) {
                                    if (Numerics.floor(lval[xyz+best*naxyz]/100.0f)==obj1+1) {
                                       heap.addValue(pval[ngb+best*naxyz],ngb,lval[xyz+best*naxyz]);
                                       best=nbest;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return labels;            
	}
			
	public void conditionalVolumeCertaintyThreshold(float spread) {
	    // main idea: region growing from inside, until within volume prior
	    // and a big enough difference in "certainty" score?
	    
		// find appropriate threshold to have correct volume; should use a fast marching approach!
		BinaryHeapPair	heap = new BinaryHeapPair(nx*ny+ny*nz+nz*nx, BinaryHeapPair.MAXTREE);
		int[] labels = new int[ndata];
        int[] start = new int[nobj];
        float[] bestscore = new float[nobj];
		for (int obj=1;obj<nobj;obj++) bestscore[obj] = -INF;
		double[] voldata = new double[nobj];
		double[] avgbound = new double[nobj];
        double[] devbound = new double[nobj];
        double[] devdiff = new double[nobj];
        int[] nbound = new int[nobj];
        int[][] nextbest = new int[nobj][ndata];
		heap.reset();
		
		// first, find where then next objects probability is in the stack
		for (int obj=1;obj<nobj;obj++) {
		    for (int n=0;n<ndata;n++) {
		        nextbest[obj][n] = 0;
		        if (combinedLabels[0][n]>100*(obj+1) && combinedLabels[0][n]<100*(obj+2)) {
		            // find the highest proba for a different structure
                    for (int b=1;b<nbest;b++) {
                        if (combinedLabels[b][n]<100*(obj+1) || combinedLabels[b][n]>100*(obj+2)) {
                            nextbest[obj][n] = b;
                            b = nbest;
                        }
                    }
                } else {
                    // find the highest proba for current structure
                    for (int b=1;b<nbest;b++) {
                        if (combinedLabels[b][n]>100*(obj+1) || combinedLabels[b][n]<100*(obj+2)) {
                            nextbest[obj][n] = -b;
                            b = nbest;
                        }
                    }
                }
            }
        }
		// important: skip first label as background (allows for unbounded growth)
        for (int obj=1;obj<nobj;obj++) {
		    // find highest scoring voxel as starting point
           for (int b=0;b<nbest;b++) {
               for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
                    int xyz=x+nx*y+nx*ny*z;
                    if (mask[xyz]) {
                        int id = idmap[xyz];
                        if (combinedLabels[b][id]>100*(obj+1) && combinedLabels[b][id]<100*(obj+2)) {
                        //if (combinedLabels[b][idmap[xyz]]==101*(obj+1)) {
                            float score;
                            if (b==0) score = combinedProbas[0][id]-combinedProbas[nextbest[obj][id]][id];
                            else score = combinedProbas[b][id]-combinedProbas[0][id];
                            //score = combinedProbas[b][idmap[xyz]];
                            if (score>bestscore[obj]) {
                                bestscore[obj] = score;
                                start[obj] = xyz;
                            }
                            if (b==0) voldata[obj] += rx*ry*rz;
                        }
                    }
                }
                if (bestscore[obj]>-INF) b = nbest;
            }
            heap.addValue(bestscore[obj],start[obj],101*(obj+1));
            
            // boundary: mean difference
            for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
                int xyz=x+nx*y+nx*ny*z;
                if (mask[xyz]) {
                    if ((combinedLabels[0][idmap[xyz]]>100*(obj+1) && combinedLabels[0][idmap[xyz]]<100*(obj+2)) || start[obj]==xyz) {
                        for (byte k = 0; k<26; k++) {
                            int ngb = Ngb.neighborIndex(k, xyz, nx, ny, nz);
                            if (mask[ngb]) {
                                // also measure the neighbors' values -> count every pair (values may be used multiple times,
                                // but the number of samples is equal for inside and outside)
                                if (combinedLabels[0][idmap[ngb]]<100*(obj+1) || combinedLabels[0][idmap[ngb]]>100*(obj+2)) {
                                    avgbound[obj] += combinedProbas[0][idmap[xyz]];
                                    avgbound[obj] += combinedProbas[1][idmap[ngb]];
                                    nbound[obj]+=2;
                                }
                            }
                        }
                    }
                }
            }
            if (nbound[obj]>0) avgbound[obj] /= (double)nbound[obj];
            // boundary: stdev difference
            for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
                int xyz=x+nx*y+nx*ny*z;
                if (mask[xyz]) {
                    if ((combinedLabels[0][idmap[xyz]]>100*(obj+1) && combinedLabels[0][idmap[xyz]]<100*(obj+2)) || start[obj]==xyz) {
                        for (byte k = 0; k<26; k++) {
                            int ngb = Ngb.neighborIndex(k, xyz, nx, ny, nz);
                            if (mask[ngb]) {
                                if (combinedLabels[0][idmap[ngb]]<100*(obj+1) || combinedLabels[0][idmap[ngb]]>100*(obj+2)) {
                                    devbound[obj] += Numerics.square(combinedProbas[0][idmap[xyz]]-avgbound[obj]);
                                    devbound[obj] += Numerics.square(combinedProbas[1][idmap[ngb]]-avgbound[obj]);
                                    if (nextbest[obj][idmap[xyz]]>0)
                                        devdiff[obj] += Numerics.square(combinedProbas[0][idmap[xyz]]-combinedProbas[nextbest[obj][idmap[xyz]]][idmap[xyz]]);
                                    else
                                        devdiff[obj] += Numerics.square(combinedProbas[-nextbest[obj][idmap[xyz]]][idmap[xyz]]-combinedProbas[0][idmap[xyz]]);
                                    if (nextbest[obj][idmap[ngb]]<0)
                                        devdiff[obj] += Numerics.square(combinedProbas[-nextbest[obj][idmap[ngb]]][idmap[ngb]]-combinedProbas[0][idmap[ngb]]);
                                    else
                                        devdiff[obj] += Numerics.square(combinedProbas[0][idmap[ngb]]-combinedProbas[nextbest[obj][idmap[ngb]]][idmap[ngb]]);
                                    
                                }
                            }
                        }
                    }
                }
            }
            if (nbound[obj]>1) {
                devbound[obj] /= (nbound[obj]-1.0);
                devdiff[obj] /= (nbound[obj]-1.0);
            }
        }
        // Posterior volumes:
        for (int obj=1;obj<nobj;obj++) {
            double logvolmean = 0.5*logVolMean[obj]+0.5*FastMath.log(voldata[obj]);
            double logvolstdv = FastMath.sqrt( 0.5*( Numerics.square(logVolStdv[obj])
                                + 0.5*Numerics.square(logVolMean[obj]-FastMath.log(voldata[obj])) ) );
            System.out.print("Label "+obj+": atlas volume = "+FastMath.exp(logVolMean[obj])+" ["+FastMath.exp(logVolMean[obj]-logVolStdv[obj])+", "+FastMath.exp(logVolMean[obj]+logVolStdv[obj])+"]");
            System.out.print(", data volume: "+voldata[obj]+" -> posterior volume = "+FastMath.exp(logvolmean)+" ["+FastMath.exp(logvolmean-logvolstdv)+", "+FastMath.exp(logvolmean+logvolstdv)+"]\n");
            logVolMean[obj] = (float)logvolmean;
            logVolStdv[obj] = (float)logvolstdv;
        }   
        // Boundary statistics
        for (int obj=1;obj<nobj;obj++) {
            System.out.print("Label "+obj+": boundary = "+avgbound[obj]+" +/- "+FastMath.sqrt(devbound[obj])+" (difference: "+FastMath.sqrt(devdiff[obj])+")\n");
        }   
                
        float[] prev = new float[nobj];
        double[] vol = new double[nobj];
        double[] bestvol = new double[nobj];
        double[] bestproba = new double[nobj];
        while (heap.isNotEmpty()) {
            float score = heap.getFirst();
            int xyz = heap.getFirstId1();
            int obj1obj2 = heap.getFirstId2();
            heap.removeFirst();
            if (labels[idmap[xyz]]==0) {
                int obj = Numerics.floor(obj1obj2/100)-1;
                // update the values
                vol[obj]+= rx*ry*rz;
                labels[idmap[xyz]] = obj;
                prev[obj] = score;
                
                // compute the joint probability function
                double pvol = FastMath.exp(-0.5*(FastMath.log(Numerics.max(1.0,vol[obj]))-logVolMean[obj])
                                               *(FastMath.log(Numerics.max(1.0,vol[obj]))-logVolMean[obj])
                                               /Numerics.max(1.0,(logVolStdv[obj]*logVolStdv[obj])));
                //double pdiff = 1.0-FastMath.exp(-0.5*(score-prev[obj])*(score-prev[obj])/(scale*scale));
                //double pcert = FastMath.exp(-0.5*(score-avgbound[obj])*(score-avgbound[obj])/devbound[obj]);
                double pcert = FastMath.exp(-0.5*(score*score)/devdiff[obj]);
                
                /* not the most meaningful, if score is driving the growth
                // find the proba, not the certainty?
                double proba = combinedProbas[0][idmap[xyz]];
                if (combinedLabels[0][idmap[xyz]]!=obj1obj2) {
                    for (int b=1;b<nbest;b++) {
                        if (combinedLabels[b][idmap[xyz]]==obj1obj2) {
                            proba = combinedProbas[b][idmap[xyz]];
                            b=nbest;
                        }
                    }
                }
                double pprob = FastMath.exp(-0.5*(proba-avgbound[obj])*(proba-avgbound[obj])/devbound[obj]);
                */
                //double pstop = pvol*pcert*pprob;
                //double pstop = pvol*pprob;
                double pstop = pvol*pcert;
                
                if (pstop>bestproba[obj] && vol[obj]>FastMath.exp(logVolMean[obj]-spread*logVolStdv[obj])) {
                    bestproba[obj] = pstop;
                    bestvol[obj] = vol[obj];
                }
                // run until the volume exceeds the mean volume + n*stdev
                if (vol[obj]<FastMath.exp(logVolMean[obj]+spread*logVolStdv[obj])) {
                    // add neighbors
                    //for (byte k = 0; k<6; k++) {
                    for (byte k = 0; k<26; k++) {
                        int ngb = Ngb.neighborIndex(k, xyz, nx, ny, nz);
                        if (ngb>0 && ngb<nxyz && idmap[ngb]>-1) {
                            if (mask[ngb]) {
                                if (labels[idmap[ngb]]==0) {
                                    for (int best=0;best<nbest;best++) {
                                        // same label
                                        if (combinedLabels[best][idmap[ngb]]==obj1obj2) {
                                            if (best==0) {
                                                heap.addValue(combinedProbas[0][idmap[ngb]]-combinedProbas[nextbest[obj][idmap[ngb]]][idmap[ngb]],ngb,obj1obj2);
                                            } else {
                                                if (nextbest[obj][idmap[ngb]]>0) {
                                                    heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[nextbest[obj][idmap[ngb]]][idmap[ngb]],ngb,obj1obj2);
                                                } else {
                                                    heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[0][idmap[ngb]],ngb,obj1obj2);
                                                }
                                            }
                                            best=nbest;
                                        } else 
                                        // from object to conditional boundary label
                                        if (obj1obj2==101*(obj+1) && combinedLabels[best][idmap[ngb]]>100*(obj+1) && combinedLabels[best][idmap[ngb]]<100*(obj+2)) {
                                            if (best==0) {
                                                heap.addValue(combinedProbas[0][idmap[ngb]]-combinedProbas[nextbest[obj][idmap[ngb]]][idmap[ngb]],ngb,combinedLabels[best][idmap[ngb]]);
                                            } else {
                                                if (nextbest[obj][idmap[ngb]]>0) {
                                                    heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[nextbest[obj][idmap[ngb]]][idmap[ngb]],ngb,combinedLabels[best][idmap[ngb]]);
                                                } else {
                                                    heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[0][idmap[ngb]],ngb,obj1obj2);
                                                }
                                            }
                                            best=nbest;
                                        } else
                                        // from conditional boundary to conditional boundary
                                        if (obj1obj2>100*(obj+1) && obj1obj2<100*(obj+2) && combinedLabels[best][idmap[ngb]]>100*(obj+1) && combinedLabels[best][idmap[ngb]]<100*(obj+2)) {
                                            if (best==0) {
                                                heap.addValue(combinedProbas[0][idmap[ngb]]-combinedProbas[nextbest[obj][idmap[ngb]]][idmap[ngb]],ngb,combinedLabels[best][idmap[ngb]]);
                                            } else {
                                                if (nextbest[obj][idmap[ngb]]>0) {
                                                    heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[nextbest[obj][idmap[ngb]]][idmap[ngb]],ngb,combinedLabels[best][idmap[ngb]]);
                                                } else {
                                                    heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[0][idmap[ngb]],ngb,obj1obj2);
                                                }
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
        }
        System.out.println("\nOptimized volumes: ");
        for (int obj=1;obj<nobj;obj++) System.out.println(obj+": "+bestvol[obj]+" ("+bestproba[obj]+") ");
        // re-run one last time to get the segmentation
        heap.reset();
        for (int obj=0;obj<nobj;obj++) {
            vol[obj] = 0.0;
        }
        for(int id=0;id<ndata;id++) labels[id] = 0;
        for (int obj=1;obj<nobj;obj++) {
            heap.addValue(bestscore[obj],start[obj],101*(obj+1));
        }
        while (heap.isNotEmpty()) {
            float score = heap.getFirst();
            int xyz = heap.getFirstId1();
            int obj1obj2 = heap.getFirstId2();
            heap.removeFirst();
            if (labels[idmap[xyz]]==0) {
                int obj = Numerics.floor(obj1obj2/100)-1;
                if (vol[obj]<bestvol[obj]) {
                    // update the values
                    vol[obj]+=rx*ry*rz;
                    labels[idmap[xyz]] = obj;
                
                    // add neighbors
                    //for (byte k = 0; k<6; k++) {
                    for (byte k = 0; k<26; k++) {
                        int ngb = Ngb.neighborIndex(k, xyz, nx, ny, nz);
                        if (ngb>0 && ngb<nxyz && idmap[ngb]>-1) {
                            if (mask[ngb]) {
                                if (labels[idmap[ngb]]==0) {
                                    for (int best=0;best<nbest;best++) {
                                        // same label
                                        if (combinedLabels[best][idmap[ngb]]==obj1obj2) {
                                            if (best==0) {
                                                heap.addValue(combinedProbas[0][idmap[ngb]]-combinedProbas[nextbest[obj][idmap[ngb]]][idmap[ngb]],ngb,obj1obj2);
                                            } else {
                                                if (nextbest[obj][idmap[ngb]]>0) {
                                                    heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[nextbest[obj][idmap[ngb]]][idmap[ngb]],ngb,obj1obj2);
                                                } else {
                                                    heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[0][idmap[ngb]],ngb,obj1obj2);
                                                }
                                            }
                                            best=nbest;
                                        } else 
                                        // from object to conditional boundary label
                                        if (obj1obj2==101*(obj+1) && combinedLabels[best][idmap[ngb]]>100*(obj+1) && combinedLabels[best][idmap[ngb]]<100*(obj+2)) {
                                            if (best==0) {
                                                heap.addValue(combinedProbas[0][idmap[ngb]]-combinedProbas[nextbest[obj][idmap[ngb]]][idmap[ngb]],ngb,combinedLabels[best][idmap[ngb]]);
                                            } else {
                                                if (nextbest[obj][idmap[ngb]]>0) {
                                                    heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[nextbest[obj][idmap[ngb]]][idmap[ngb]],ngb,combinedLabels[best][idmap[ngb]]);
                                                } else {
                                                    heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[0][idmap[ngb]],ngb,obj1obj2);
                                                }
                                            }
                                            best=nbest;
                                        } else
                                        // from conditional boundary to conditional boundary
                                        if (obj1obj2>100*(obj+1) && obj1obj2<100*(obj+2) && combinedLabels[best][idmap[ngb]]>100*(obj+1) && combinedLabels[best][idmap[ngb]]<100*(obj+2)) {
                                            if (best==0) {
                                                heap.addValue(combinedProbas[0][idmap[ngb]]-combinedProbas[nextbest[obj][idmap[ngb]]][idmap[ngb]],ngb,combinedLabels[best][idmap[ngb]]);
                                            } else {
                                                if (nextbest[obj][idmap[ngb]]>0) {
                                                    heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[nextbest[obj][idmap[ngb]]][idmap[ngb]],ngb,combinedLabels[best][idmap[ngb]]);
                                                } else {
                                                    heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[0][idmap[ngb]],ngb,obj1obj2);
                                                }
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
        }
        // final segmentation: collapse onto result images
        finalLabel = new int[nxyz];
        finalProba = new float[nxyz];
        float[] kept = new float[nobj];
        for (int x=1;x<ntx-1;x++) for (int y=1;y<nty-1;y++) for (int z=1;z<ntz-1;z++) {
            int xyz = x+ntx*y+ntx*nty*z;
            if (mask[xyz]) {
                int obj = labels[idmap[xyz]];
                for (int best=0;best<nbest;best++) {
                    if (combinedLabels[best][idmap[xyz]]>100*(obj+1) && combinedLabels[best][idmap[xyz]]<100*(obj+2)) {
                        /*if (best==0) {
                            finalProba[xyz] = combinedProbas[0][idmap[xyz]]-combinedProbas[1][idmap[xyz]];
                        } else {
                            finalProba[xyz] = combinedProbas[best][idmap[xyz]]-combinedProbas[0][idmap[xyz]];
                        }*/
                        finalProba[xyz] = combinedProbas[best][idmap[xyz]];
                        best=nbest;
                    }
                }
                finalLabel[xyz] = obj;
            }
        }
        return;            
	}
	
	public void conditionalBoundaryVolumeCertaintyThreshold(float spread) {
	    // main idea: region growing from inside, until within volume prior
	    // and a big enough difference in "certainty" score?
	    
		// find appropriate threshold to have correct volume; should use a fast marching approach!
		BinaryHeapPair	heap = new BinaryHeapPair(nx*ny+ny*nz+nz*nx, BinaryHeapPair.MAXTREE);
		int[] labels = new int[ndata];
        int[] start = new int[nobj];
        float[] bestscore = new float[nobj];
		for (int obj=1;obj<nobj;obj++) bestscore[obj] = -INF;
		double[][] voldata = new double[nobj][nobj];
		double[] avgbound = new double[nobj];
        double[] devbound = new double[nobj];
        double[] devdiff = new double[nobj];
        int[] nbound = new int[nobj];
        int[][] nextbest = new int[nobj][ndata];
		heap.reset();
		
		// first, find where then next objects probability is in the stack
		for (int obj=1;obj<nobj;obj++) {
		    for (int n=0;n<ndata;n++) {
		        nextbest[obj][n] = 0;
		        if (combinedLabels[0][n]>100*(obj+1) && combinedLabels[0][n]<100*(obj+2)) {
		            // find the highest proba for a different structure
                    for (int b=1;b<nbest;b++) {
                        if (combinedLabels[b][n]<100*(obj+1) || combinedLabels[b][n]>100*(obj+2)) {
                            nextbest[obj][n] = b;
                            b = nbest;
                        }
                    }
                } else {
                    // find the highest proba for current structure
                    for (int b=1;b<nbest;b++) {
                        if (combinedLabels[b][n]>100*(obj+1) || combinedLabels[b][n]<100*(obj+2)) {
                            nextbest[obj][n] = -b;
                            b = nbest;
                        }
                    }
                }
            }
        }
		// important: skip first label as background (allows for unbounded growth)
        for (int obj=1;obj<nobj;obj++) {
		    // find highest scoring voxel as starting point
           for (int b=0;b<nbest;b++) {
               for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
                    int xyz=x+nx*y+nx*ny*z;
                    if (mask[xyz]) {
                        int id = idmap[xyz];
                        if (combinedLabels[b][id]>100*(obj+1) && combinedLabels[b][id]<100*(obj+2)) {
                        //if (combinedLabels[b][idmap[xyz]]==101*(obj+1)) {
                            float score;
                            if (b==0) score = combinedProbas[0][id]-combinedProbas[nextbest[obj][id]][id];
                            else score = combinedProbas[b][id]-combinedProbas[0][id];
                            //score = combinedProbas[b][idmap[xyz]];
                            if (score>bestscore[obj]) {
                                bestscore[obj] = score;
                                start[obj] = xyz;
                            }
                        }
                    }
                }
                if (bestscore[obj]>-INF) b = nbest;
            }
            heap.addValue(bestscore[obj],start[obj],101*(obj+1));
            
            // boundary: mean difference
            for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
                int xyz=x+nx*y+nx*ny*z;
                if (mask[xyz]) {
                    if ((combinedLabels[0][idmap[xyz]]>100*(obj+1) && combinedLabels[0][idmap[xyz]]<100*(obj+2)) || start[obj]==xyz) {
                        for (byte k = 0; k<26; k++) {
                            int ngb = Ngb.neighborIndex(k, xyz, nx, ny, nz);
                            if (mask[ngb]) {
                                // also measure the neighbors' values -> count every pair (values may be used multiple times,
                                // but the number of samples is equal for inside and outside)
                                if (combinedLabels[0][idmap[ngb]]<100*(obj+1) || combinedLabels[0][idmap[ngb]]>100*(obj+2)) {
                                    avgbound[obj] += combinedProbas[0][idmap[xyz]];
                                    avgbound[obj] += combinedProbas[1][idmap[ngb]];
                                    nbound[obj]+=2;
                                }
                            }
                        }
                    }
                }
            }
            if (nbound[obj]>0) avgbound[obj] /= (double)nbound[obj];
            // boundary: stdev difference
            for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
                int xyz=x+nx*y+nx*ny*z;
                if (mask[xyz]) {
                    if ((combinedLabels[0][idmap[xyz]]>100*(obj+1) && combinedLabels[0][idmap[xyz]]<100*(obj+2)) || start[obj]==xyz) {
                        for (byte k = 0; k<26; k++) {
                            int ngb = Ngb.neighborIndex(k, xyz, nx, ny, nz);
                            if (mask[ngb]) {
                                if (combinedLabels[0][idmap[ngb]]<100*(obj+1) || combinedLabels[0][idmap[ngb]]>100*(obj+2)) {
                                    devbound[obj] += Numerics.square(combinedProbas[0][idmap[xyz]]-avgbound[obj]);
                                    devbound[obj] += Numerics.square(combinedProbas[1][idmap[ngb]]-avgbound[obj]);
                                    if (nextbest[obj][idmap[xyz]]>0)
                                        devdiff[obj] += Numerics.square(combinedProbas[0][idmap[xyz]]-combinedProbas[nextbest[obj][idmap[xyz]]][idmap[xyz]]);
                                    else
                                        devdiff[obj] += Numerics.square(combinedProbas[-nextbest[obj][idmap[xyz]]][idmap[xyz]]-combinedProbas[0][idmap[xyz]]);
                                    if (nextbest[obj][idmap[ngb]]<0)
                                        devdiff[obj] += Numerics.square(combinedProbas[-nextbest[obj][idmap[ngb]]][idmap[ngb]]-combinedProbas[0][idmap[ngb]]);
                                    else
                                        devdiff[obj] += Numerics.square(combinedProbas[0][idmap[ngb]]-combinedProbas[nextbest[obj][idmap[ngb]]][idmap[ngb]]);
                                    
                                }
                            }
                        }
                    }
                }
            }
            if (nbound[obj]>1) {
                devbound[obj] /= (nbound[obj]-1.0);
                devdiff[obj] /= (nbound[obj]-1.0);
            }
        }
        // estimate voxel prior volumes
        for (int obj1=1;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
            for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
                int xyz=x+nx*y+nx*ny*z;
                if (mask[xyz]) {
                    int id = idmap[xyz];
                    if (combinedLabels[0][id]==100*(obj1+1)+(obj2+1)) {
                        voldata[obj1][obj2] += rx*ry*rz;
                    }
                }
            }
        }
        
        /*
        // Posterior volumes:
        for (int obj=1;obj<nobj;obj++) {
            double logvolmean = 0.5*logVolMean[obj]+0.5*FastMath.log(voldata[obj]);
            double logvolstdv = FastMath.sqrt( 0.5*( Numerics.square(logVolStdv[obj])
                                + 0.5*Numerics.square(logVolMean[obj]-FastMath.log(voldata[obj])) ) );
            System.out.print("Label "+obj+": atlas volume = "+FastMath.exp(logVolMean[obj])+" ["+FastMath.exp(logVolMean[obj]-logVolStdv[obj])+", "+FastMath.exp(logVolMean[obj]+logVolStdv[obj])+"]");
            System.out.print(", data volume: "+voldata[obj]+" -> posterior volume = "+FastMath.exp(logvolmean)+" ["+FastMath.exp(logvolmean-logvolstdv)+", "+FastMath.exp(logvolmean+logvolstdv)+"]\n");
            logVolMean[obj] = (float)logvolmean;
            logVolStdv[obj] = (float)logvolstdv;
        }   
        */
        // Posterior volumes:
        for (int obj1=1;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
            double logvolmean = 0.5*logVolMean2[obj1][obj2]+0.5*FastMath.log(Numerics.max(1.0,voldata[obj1][obj2]));
            double logvolstdv = FastMath.sqrt( 0.5*( Numerics.square(logVolStdv2[obj1][obj2])
                                + 0.5*Numerics.square(logVolMean2[obj1][obj2]-FastMath.log(Numerics.max(1.0,voldata[obj1][obj2]))) ) );
            System.out.print("Label "+obj1+" | "+obj2+": atlas volume = "+FastMath.exp(logVolMean2[obj1][obj2])+" ["+FastMath.exp(logVolMean2[obj1][obj2]-logVolStdv2[obj1][obj2])+", "+FastMath.exp(logVolMean2[obj1][obj2]+logVolStdv2[obj1][obj2])+"]");
            System.out.print(", data volume: "+voldata[obj1][obj2]+" -> posterior volume = "+FastMath.exp(logvolmean)+" ["+FastMath.exp(logvolmean-logvolstdv)+", "+FastMath.exp(logvolmean+logvolstdv)+"]\n");
            logVolMean2[obj1][obj2] = (float)logvolmean;
            logVolStdv2[obj1][obj2] = (float)logvolstdv;
        }   
        // Boundary statistics
        for (int obj=1;obj<nobj;obj++) {
            System.out.print("Label "+obj+": boundary = "+avgbound[obj]+" +/- "+FastMath.sqrt(devbound[obj])+" (difference: "+FastMath.sqrt(devdiff[obj])+")\n");
        }   
                
        float[] prev = new float[nobj];
        double[] vol = new double[nobj];
        double[][] vol2 = new double[nobj][nobj];
        double[] bestvol = new double[nobj];
        double[] bestproba = new double[nobj];
        while (heap.isNotEmpty()) {
            float score = heap.getFirst();
            int xyz = heap.getFirstId1();
            int obj1obj2 = heap.getFirstId2();
            heap.removeFirst();
            if (labels[idmap[xyz]]==0) {
                int obj1 = Numerics.floor(obj1obj2/100)-1;
                int obj2 = obj1obj2 - 100*(obj1+1) - 1;
                // update the values
                vol[obj1]+= rx*ry*rz;
                vol2[obj1][obj2] += rx*ry*rz;
                labels[idmap[xyz]] = obj1;
                prev[obj1] = score;
                
                // compute the joint probability function
                double pvol = 1.0;
                for (int obj=0;obj<nobj;obj++) if (logVolMean2[obj1][obj]>0) {
                    pvol *= FastMath.exp(-0.5*(FastMath.log(Numerics.max(1.0,vol2[obj1][obj]))-logVolMean2[obj1][obj])
                                               *(FastMath.log(Numerics.max(1.0,vol2[obj1][obj]))-logVolMean2[obj1][obj])
                                               /Numerics.max(1.0,(logVolStdv2[obj1][obj]*logVolStdv2[obj1][obj])));
                }
                double pcert = FastMath.exp(-0.5*(score*score)/devdiff[obj1]);
                
                double pstop = pvol*pcert;
                
                if (pstop>bestproba[obj1] && vol[obj1]>FastMath.exp(logVolMean[obj1]-spread*logVolStdv[obj1])) {
                    bestproba[obj1] = pstop;
                    bestvol[obj1] = vol[obj1];
                }
                // run until the volume exceeds the mean volume + n*stdev
                if (vol[obj1]<FastMath.exp(logVolMean[obj1]+spread*logVolStdv[obj1])) {
                    // add neighbors
                    //for (byte k = 0; k<6; k++) {
                    for (byte k = 0; k<26; k++) {
                        int ngb = Ngb.neighborIndex(k, xyz, nx, ny, nz);
                        if (ngb>0 && ngb<nxyz && idmap[ngb]>-1) {
                            if (mask[ngb]) {
                                if (labels[idmap[ngb]]==0) {
                                    for (int best=0;best<nbest;best++) {
                                        // same label
                                        if (combinedLabels[best][idmap[ngb]]==obj1obj2) {
                                            if (best==0) {
                                                heap.addValue(combinedProbas[0][idmap[ngb]]-combinedProbas[nextbest[obj1][idmap[ngb]]][idmap[ngb]],ngb,obj1obj2);
                                            } else {
                                                if (nextbest[obj1][idmap[ngb]]>0) {
                                                    heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[nextbest[obj1][idmap[ngb]]][idmap[ngb]],ngb,obj1obj2);
                                                } else {
                                                    heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[0][idmap[ngb]],ngb,obj1obj2);
                                                }
                                            }
                                            best=nbest;
                                        } else 
                                        // from object to conditional boundary label
                                        if (obj1obj2==101*(obj1+1) && combinedLabels[best][idmap[ngb]]>100*(obj1+1) && combinedLabels[best][idmap[ngb]]<100*(obj1+2)) {
                                            if (best==0) {
                                                heap.addValue(combinedProbas[0][idmap[ngb]]-combinedProbas[nextbest[obj1][idmap[ngb]]][idmap[ngb]],ngb,combinedLabels[best][idmap[ngb]]);
                                            } else {
                                                if (nextbest[obj1][idmap[ngb]]>0) {
                                                    heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[nextbest[obj1][idmap[ngb]]][idmap[ngb]],ngb,combinedLabels[best][idmap[ngb]]);
                                                } else {
                                                    heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[0][idmap[ngb]],ngb,obj1obj2);
                                                }
                                            }
                                            best=nbest;
                                        } else
                                        // from conditional boundary to conditional boundary
                                        if (obj1obj2>100*(obj1+1) && obj1obj2<100*(obj1+2) && combinedLabels[best][idmap[ngb]]>100*(obj1+1) && combinedLabels[best][idmap[ngb]]<100*(obj1+2)) {
                                            if (best==0) {
                                                heap.addValue(combinedProbas[0][idmap[ngb]]-combinedProbas[nextbest[obj1][idmap[ngb]]][idmap[ngb]],ngb,combinedLabels[best][idmap[ngb]]);
                                            } else {
                                                if (nextbest[obj1][idmap[ngb]]>0) {
                                                    heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[nextbest[obj1][idmap[ngb]]][idmap[ngb]],ngb,combinedLabels[best][idmap[ngb]]);
                                                } else {
                                                    heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[0][idmap[ngb]],ngb,obj1obj2);
                                                }
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
        }
        System.out.println("\nOptimized volumes: ");
        for (int obj=1;obj<nobj;obj++) System.out.println(obj+": "+bestvol[obj]+" ("+bestproba[obj]+") ");
        // re-run one last time to get the segmentation
        heap.reset();
        for (int obj=0;obj<nobj;obj++) {
            vol[obj] = 0.0;
        }
        for(int id=0;id<ndata;id++) labels[id] = 0;
        for (int obj=1;obj<nobj;obj++) {
            heap.addValue(bestscore[obj],start[obj],101*(obj+1));
        }
        while (heap.isNotEmpty()) {
            float score = heap.getFirst();
            int xyz = heap.getFirstId1();
            int obj1obj2 = heap.getFirstId2();
            heap.removeFirst();
            if (labels[idmap[xyz]]==0) {
                int obj = Numerics.floor(obj1obj2/100)-1;
                if (vol[obj]<bestvol[obj]) {
                    // update the values
                    vol[obj]+=rx*ry*rz;
                    labels[idmap[xyz]] = obj;
                
                    // add neighbors
                    //for (byte k = 0; k<6; k++) {
                    for (byte k = 0; k<26; k++) {
                        int ngb = Ngb.neighborIndex(k, xyz, nx, ny, nz);
                        if (ngb>0 && ngb<nxyz && idmap[ngb]>-1) {
                            if (mask[ngb]) {
                                if (labels[idmap[ngb]]==0) {
                                    for (int best=0;best<nbest;best++) {
                                        // same label
                                        if (combinedLabels[best][idmap[ngb]]==obj1obj2) {
                                            if (best==0) {
                                                heap.addValue(combinedProbas[0][idmap[ngb]]-combinedProbas[nextbest[obj][idmap[ngb]]][idmap[ngb]],ngb,obj1obj2);
                                            } else {
                                                if (nextbest[obj][idmap[ngb]]>0) {
                                                    heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[nextbest[obj][idmap[ngb]]][idmap[ngb]],ngb,obj1obj2);
                                                } else {
                                                    heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[0][idmap[ngb]],ngb,obj1obj2);
                                                }
                                            }
                                            best=nbest;
                                        } else 
                                        // from object to conditional boundary label
                                        if (obj1obj2==101*(obj+1) && combinedLabels[best][idmap[ngb]]>100*(obj+1) && combinedLabels[best][idmap[ngb]]<100*(obj+2)) {
                                            if (best==0) {
                                                heap.addValue(combinedProbas[0][idmap[ngb]]-combinedProbas[nextbest[obj][idmap[ngb]]][idmap[ngb]],ngb,combinedLabels[best][idmap[ngb]]);
                                            } else {
                                                if (nextbest[obj][idmap[ngb]]>0) {
                                                    heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[nextbest[obj][idmap[ngb]]][idmap[ngb]],ngb,combinedLabels[best][idmap[ngb]]);
                                                } else {
                                                    heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[0][idmap[ngb]],ngb,obj1obj2);
                                                }
                                            }
                                            best=nbest;
                                        } else
                                        // from conditional boundary to conditional boundary
                                        if (obj1obj2>100*(obj+1) && obj1obj2<100*(obj+2) && combinedLabels[best][idmap[ngb]]>100*(obj+1) && combinedLabels[best][idmap[ngb]]<100*(obj+2)) {
                                            if (best==0) {
                                                heap.addValue(combinedProbas[0][idmap[ngb]]-combinedProbas[nextbest[obj][idmap[ngb]]][idmap[ngb]],ngb,combinedLabels[best][idmap[ngb]]);
                                            } else {
                                                if (nextbest[obj][idmap[ngb]]>0) {
                                                    heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[nextbest[obj][idmap[ngb]]][idmap[ngb]],ngb,combinedLabels[best][idmap[ngb]]);
                                                } else {
                                                    heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[0][idmap[ngb]],ngb,obj1obj2);
                                                }
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
        }
        // final segmentation: collapse onto result images
        finalLabel = new int[nxyz];
        finalProba = new float[nxyz];
        float[] kept = new float[nobj];
        for (int x=1;x<ntx-1;x++) for (int y=1;y<nty-1;y++) for (int z=1;z<ntz-1;z++) {
            int xyz = x+ntx*y+ntx*nty*z;
            if (mask[xyz]) {
                int obj = labels[idmap[xyz]];
                for (int best=0;best<nbest;best++) {
                    if (combinedLabels[best][idmap[xyz]]>100*(obj+1) && combinedLabels[best][idmap[xyz]]<100*(obj+2)) {
                        /*if (best==0) {
                            finalProba[xyz] = combinedProbas[0][idmap[xyz]]-combinedProbas[1][idmap[xyz]];
                        } else {
                            finalProba[xyz] = combinedProbas[best][idmap[xyz]]-combinedProbas[0][idmap[xyz]];
                        }*/
                        finalProba[xyz] = combinedProbas[best][idmap[xyz]];
                        best=nbest;
                    }
                }
                finalLabel[xyz] = obj;
            }
        }
        return;            
	}
	
	public void conditionalBoundaryGrowth(float spread) {
	    // main idea: region growing from inside, until within volume prior
	    // and a big enough difference in "certainty" score?
	    
		// find appropriate threshold to have correct volume; should use a fast marching approach!
		BinaryHeapPair	heap = new BinaryHeapPair(nx*ny+ny*nz+nz*nx, BinaryHeapPair.MAXTREE);
		int[] labels = new int[ndata];
        int[][] start = new int[nobj][nobj];
        float[][] bestscore = new float[nobj][nobj];
		for (int obj1=1;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) bestscore[obj1][obj2] = -INF;
		double[][] voldata = new double[nobj][nobj];
		double[][] devdiff = new double[nobj][nobj];
        int[][] nbound = new int[nobj][nobj];
        heap.reset();
		
		// important: skip first label as background (allows for unbounded growth)
        for (int obj1=1;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
		    // find highest scoring voxel as starting point
           for (int b=0;b<nbest;b++) {
               for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
                    int xyz=x+nx*y+nx*ny*z;
                    if (mask[xyz]) {
                        int id = idmap[xyz];
                        if (combinedLabels[b][id]==100*(obj1+1)+(obj2+1)) {
                            float score;
                            if (b==0) score = combinedProbas[0][id]-combinedProbas[1][id];
                            else score = combinedProbas[b][id]-combinedProbas[0][id];
                            //score = combinedProbas[b][idmap[xyz]];
                            if (score>bestscore[obj1][obj2]) {
                                bestscore[obj1][obj2] = score;
                                start[obj1][obj2] = xyz;
                            }
                            if (b==0) voldata[obj1][obj2] += rx*ry*rz;
                        }
                    }
                }
                if (bestscore[obj1][obj2]>-INF) b = nbest;
            }
            heap.addValue(bestscore[obj1][obj2],start[obj1][obj2],100*(obj1+1)+(obj2+1));
            
            // boundary: mean difference
            for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
                int xyz=x+nx*y+nx*ny*z;
                if (mask[xyz]) {
                    if ( (combinedLabels[0][idmap[xyz]]==100*(obj1+1)+(obj2+1) ) || start[obj1][obj2]==xyz) {
                        for (byte k = 0; k<26; k++) {
                            int ngb = Ngb.neighborIndex(k, xyz, nx, ny, nz);
                            if (mask[ngb]) {
                                // also measure the neighbors' values -> count every pair (values may be used multiple times,
                                // but the number of samples is equal for inside and outside)
                                if (combinedLabels[0][idmap[ngb]]!=100*(obj1+1)+(obj2+1) ) {
                                    devdiff[obj1][obj2] += Numerics.square(combinedProbas[0][idmap[xyz]]-combinedProbas[1][idmap[xyz]]);
                                    devdiff[obj1][obj2] += Numerics.square(combinedProbas[0][idmap[ngb]]-combinedProbas[1][idmap[ngb]]);
                                    nbound[obj1][obj2] += 2;
                                }
                            }
                        }
                    }
                }
            }
            if (nbound[obj1][obj2]>1) {
                devdiff[obj1][obj2] /= (nbound[obj1][obj2]-1.0);
            }
        }
        //just in case: propagate the difference from object anything else to empty values
        for (int obj1=1;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
            if (devdiff[obj1][obj2]==0) {
                devdiff[obj1][obj2] = devdiff[obj1][obj1];       
            }
        }
        // Posterior volumes:
        for (int obj1=1;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
            double logvolmean = 0.5*logVolMean2[obj1][obj2]+0.5*FastMath.log(Numerics.max(1.0,voldata[obj1][obj2]));
            double logvolstdv = FastMath.sqrt( 0.5*( Numerics.square(logVolStdv2[obj1][obj2])
                                + 0.5*Numerics.square(logVolMean2[obj1][obj2]-FastMath.log(Numerics.max(1.0,voldata[obj1][obj2]))) ) );
            System.out.print("Label "+obj1+" | "+obj2+": atlas volume = "+FastMath.exp(logVolMean2[obj1][obj2])+" ["+FastMath.exp(logVolMean2[obj1][obj2]-logVolStdv2[obj1][obj2])+", "+FastMath.exp(logVolMean2[obj1][obj2]+logVolStdv2[obj1][obj2])+"]");
            System.out.print(", data volume: "+voldata[obj1][obj2]+" -> posterior volume = "+FastMath.exp(logvolmean)+" ["+FastMath.exp(logvolmean-logvolstdv)+", "+FastMath.exp(logvolmean+logvolstdv)+"]\n");
            logVolMean2[obj1][obj2] = (float)logvolmean;
            logVolStdv2[obj1][obj2] = (float)logvolstdv;
        }   
        // Boundary statistics
        for (int obj1=1;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
            System.out.print("Label "+obj1+" | "+obj2+": boundary = "+FastMath.sqrt(devdiff[obj1][obj2])+")\n");
        }   
                
        float[][] prev = new float[nobj][nobj];
        double[][] vol = new double[nobj][nobj];
        double[][] bestvol = new double[nobj][nobj];
        double[][] bestproba = new double[nobj][nobj];
        while (heap.isNotEmpty()) {
            float score = heap.getFirst();
            int xyz = heap.getFirstId1();
            int obj1obj2 = heap.getFirstId2();
            heap.removeFirst();
            if (labels[idmap[xyz]]==0) {
                int obj1 = Numerics.floor(obj1obj2/100)-1;
                int obj2 = obj1obj2 - 100*(obj1+1) - 1;
                // update the values
                vol[obj1][obj2] += rx*ry*rz;
                labels[idmap[xyz]] = 100*(obj1+1)+(obj2+1);
                prev[obj1][obj2] = score;
                
                // compute the joint probability function
                double pvol = FastMath.exp(-0.5*(FastMath.log(Numerics.max(1.0,vol[obj1][obj2]))-logVolMean2[obj1][obj2])
                                               *(FastMath.log(Numerics.max(1.0,vol[obj1][obj2]))-logVolMean2[obj1][obj2])
                                               /Numerics.max(1.0,(logVolStdv2[obj1][obj2]*logVolStdv2[obj1][obj2])));
                //double pdiff = 1.0-FastMath.exp(-0.5*(score-prev[obj])*(score-prev[obj])/(scale*scale));
                //double pcert = FastMath.exp(-0.5*(score-avgbound[obj])*(score-avgbound[obj])/devbound[obj]);
                double pcert = FastMath.exp(-0.5*(score*score)/devdiff[obj1][obj2]);
                
                double pstop = pvol*pcert;
                
                if (pstop>bestproba[obj1][obj2] && vol[obj1][obj2]>FastMath.exp(logVolMean2[obj1][obj2]-spread*logVolStdv2[obj1][obj2])) {
                    bestproba[obj1][obj2] = pstop;
                    bestvol[obj1][obj2] = vol[obj1][obj2];
                }
                // run until the volume exceeds the mean volume + n*stdev
                if (vol[obj1][obj2]<FastMath.exp(logVolMean2[obj1][obj2]+spread*logVolStdv2[obj1][obj2])) {
                    // add neighbors
                    //for (byte k = 0; k<6; k++) {
                    for (byte k = 0; k<26; k++) {
                        int ngb = Ngb.neighborIndex(k, xyz, nx, ny, nz);
                        if (ngb>0 && ngb<nxyz && idmap[ngb]>-1) {
                            if (mask[ngb]) {
                                if (labels[idmap[ngb]]==0) {
                                    for (int best=0;best<nbest;best++) {
                                        // same label
                                        if (combinedLabels[best][idmap[ngb]]==obj1obj2) {
                                            if (best==0) {
                                                heap.addValue(combinedProbas[0][idmap[ngb]]-combinedProbas[1][idmap[ngb]],ngb,obj1obj2);
                                            } else {
                                                heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[0][idmap[ngb]],ngb,obj1obj2);
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
        }
        System.out.println("\nOptimized volumes: ");
        for (int obj1=1;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) System.out.println(obj1+" | "+obj2+": "+bestvol[obj1][obj2]+" ("+bestproba[obj1][obj2]+") ");
        // re-run one last time to get the segmentation
        heap.reset();
        for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
            vol[obj1][obj2] = 0.0;
        }
        for(int id=0;id<ndata;id++) labels[id] = 0;
        for (int obj1=1;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
            heap.addValue(bestscore[obj1][obj2],start[obj1][obj2],100*(obj1+1)+(obj2+1));
        }
        while (heap.isNotEmpty()) {
            float score = heap.getFirst();
            int xyz = heap.getFirstId1();
            int obj1obj2 = heap.getFirstId2();
            heap.removeFirst();
            if (labels[idmap[xyz]]==0) {
                int obj1 = Numerics.floor(obj1obj2/100)-1;
                int obj2 = obj1obj2 - 100*(obj1+1) - 1;
                if (vol[obj1][obj2]<bestvol[obj1][obj2]) {
                    // update the values
                    vol[obj1][obj2]+=rx*ry*rz;
                    labels[idmap[xyz]] = obj1obj2;
                
                    // add neighbors
                    //for (byte k = 0; k<6; k++) {
                    for (byte k = 0; k<26; k++) {
                        int ngb = Ngb.neighborIndex(k, xyz, nx, ny, nz);
                        if (ngb>0 && ngb<nxyz && idmap[ngb]>-1) {
                            if (mask[ngb]) {
                                if (labels[idmap[ngb]]==0) {
                                    for (int best=0;best<nbest;best++) {
                                        // same label
                                        if (combinedLabels[best][idmap[ngb]]==obj1obj2) {
                                            if (best==0) {
                                                heap.addValue(combinedProbas[0][idmap[ngb]]-combinedProbas[1][idmap[ngb]],ngb,obj1obj2);
                                            } else {
                                                heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[0][idmap[ngb]],ngb,obj1obj2);
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
        }
        // final segmentation: collapse onto result images
        finalLabel = new int[nxyz];
        finalProba = new float[nxyz];
        float[] kept = new float[nobj];
        for (int x=1;x<ntx-1;x++) for (int y=1;y<nty-1;y++) for (int z=1;z<ntz-1;z++) {
            int xyz = x+ntx*y+ntx*nty*z;
            if (mask[xyz]) {
                // fill in the background
                if (labels[idmap[xyz]]==0) labels[idmap[xyz]] = 101;
                int obj = Numerics.floor(labels[idmap[xyz]]/100)-1;
                for (int best=0;best<nbest;best++) {
                    if (combinedLabels[best][idmap[xyz]]==labels[idmap[xyz]]) {
                    //if (combinedLabels[best][idmap[xyz]]>100*(obj+1) && combinedLabels[best][idmap[xyz]]<100*(obj+2)) {
                        finalProba[xyz] = combinedProbas[best][idmap[xyz]];
                        best=nbest;
                    }
                }
                finalLabel[xyz] = obj;
            }
        }
        return;            
	}
		
	public void conditionalObjectBoundaryGrowth(float spread) {
	    // main idea: region growing from inside, until within volume prior
	    // and a big enough difference in "certainty" score?
	    
		// find appropriate threshold to have correct volume; should use a fast marching approach!
		BinaryHeapPair	heap = new BinaryHeapPair(nx*ny+ny*nz+nz*nx, BinaryHeapPair.MAXTREE);
		int[] labels = new int[ndata];
        int[] start = new int[nobj];
        float[] bestscore = new float[nobj];
		for (int obj=1;obj<nobj;obj++) bestscore[obj] = -INF;
		double[][] voldata = new double[nobj][nobj];
		double[][] devdiff = new double[nobj][nobj];
        int[][] nbound = new int[nobj][nobj];
        heap.reset();
		
		// important: skip first label as background (allows for unbounded growth)
        for (int obj=1;obj<nobj;obj++) {
		    // find highest scoring voxel as starting point
           for (int b=0;b<nbest;b++) {
               for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
                    int xyz=x+nx*y+nx*ny*z;
                    if (mask[xyz]) {
                        int id = idmap[xyz];
                        if (combinedLabels[b][id]==101*(obj+1)) {
                            float score;
                            if (b==0) score = combinedProbas[0][id]-combinedProbas[1][id];
                            else score = combinedProbas[b][id]-combinedProbas[0][id];
                            //score = combinedProbas[b][idmap[xyz]];
                            if (score>bestscore[obj]) {
                                bestscore[obj] = score;
                                start[obj] = xyz;
                            }
                        }
                    }
                }
                if (bestscore[obj]>-INF) b = nbest;
            }
            heap.addValue(bestscore[obj],start[obj],101*(obj+1));
        }

        // estimate voxel prior volumes
        for (int obj1=1;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
            for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
                int xyz=x+nx*y+nx*ny*z;
                if (mask[xyz]) {
                    int id = idmap[xyz];
                    if (combinedLabels[0][id]==100*(obj1+1)+(obj2+1)) {
                        voldata[obj1][obj2] += rx*ry*rz;
                    }
                }
            }
        }
        
        // boundary: mean difference
        for (int obj1=1;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
            for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
                int xyz=x+nx*y+nx*ny*z;
                if (mask[xyz]) {
                    if ( (combinedLabels[0][idmap[xyz]]==100*(obj1+1)+(obj2+1) ) || start[obj1]==xyz) {
                        for (byte k = 0; k<26; k++) {
                            int ngb = Ngb.neighborIndex(k, xyz, nx, ny, nz);
                            if (mask[ngb]) {
                                // also measure the neighbors' values -> count every pair (values may be used multiple times,
                                // but the number of samples is equal for inside and outside)
                                if (combinedLabels[0][idmap[ngb]]!=100*(obj1+1)+(obj2+1) ) {
                                    devdiff[obj1][obj2] += Numerics.square(combinedProbas[0][idmap[xyz]]-combinedProbas[1][idmap[xyz]]);
                                    devdiff[obj1][obj2] += Numerics.square(combinedProbas[0][idmap[ngb]]-combinedProbas[1][idmap[ngb]]);
                                    nbound[obj1][obj2] += 2;
                                }
                            }
                        }
                    }
                }
            }
            if (nbound[obj1][obj2]>1) {
                devdiff[obj1][obj2] /= (nbound[obj1][obj2]-1.0);
            }
        }
        //just in case: propagate the difference from object anything else to empty values
        for (int obj1=1;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
            if (devdiff[obj1][obj2]==0) {
                devdiff[obj1][obj2] = devdiff[obj1][obj1];       
            }
        }
        
        // Posterior volumes:
        for (int obj1=1;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
            double logvolmean = 0.5*logVolMean2[obj1][obj2]+0.5*FastMath.log(Numerics.max(1.0,voldata[obj1][obj2]));
            double logvolstdv = FastMath.sqrt( 0.5*( Numerics.square(logVolStdv2[obj1][obj2])
                                + 0.5*Numerics.square(logVolMean2[obj1][obj2]-FastMath.log(Numerics.max(1.0,voldata[obj1][obj2]))) ) );
            System.out.print("Label "+obj1+" | "+obj2+": atlas volume = "+FastMath.exp(logVolMean2[obj1][obj2])+" ["+FastMath.exp(logVolMean2[obj1][obj2]-logVolStdv2[obj1][obj2])+", "+FastMath.exp(logVolMean2[obj1][obj2]+logVolStdv2[obj1][obj2])+"]");
            System.out.print(", data volume: "+voldata[obj1][obj2]+" -> posterior volume = "+FastMath.exp(logvolmean)+" ["+FastMath.exp(logvolmean-logvolstdv)+", "+FastMath.exp(logvolmean+logvolstdv)+"]\n");
            logVolMean2[obj1][obj2] = (float)logvolmean;
            logVolStdv2[obj1][obj2] = (float)logvolstdv;
        }   
        // Boundary statistics
        for (int obj1=1;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
            System.out.print("Label "+obj1+" | "+obj2+": boundary = "+FastMath.sqrt(devdiff[obj1][obj2])+")\n");
        }   
                
        float[][] prev = new float[nobj][nobj];
        double[][] vol = new double[nobj][nobj];
        double[] bestvol = new double[nobj];
        double[] bestproba = new double[nobj];
        while (heap.isNotEmpty()) {
            float score = heap.getFirst();
            int xyz = heap.getFirstId1();
            int obj1obj2 = heap.getFirstId2();
            heap.removeFirst();
            if (labels[idmap[xyz]]==0) {
                int obj1 = Numerics.floor(obj1obj2/100)-1;
                int obj2 = obj1obj2 - 100*(obj1+1) - 1;
                // update volume before computing the probability
                vol[obj1][obj2] += rx*ry*rz;
                labels[idmap[xyz]] = 100*(obj1+1)+(obj2+1);
                prev[obj1][obj2] = score;
                
                // compute the joint probability function for the boundary or the whole object?
                double pvol = 1.0;
                for (int obj=0;obj<nobj;obj++) if (logVolMean2[obj1][obj]>0) {
                    pvol *= FastMath.exp(-0.5*(FastMath.log(Numerics.max(1.0,vol[obj1][obj]))-logVolMean2[obj1][obj])
                                               *(FastMath.log(Numerics.max(1.0,vol[obj1][obj]))-logVolMean2[obj1][obj])
                                               /Numerics.max(1.0,(logVolStdv2[obj1][obj]*logVolStdv2[obj1][obj])));
                }
                //double pvol = FastMath.exp(-0.5*(FastMath.log(Numerics.max(1.0,vol[obj1][obj2])-logVolMean2[obj1][obj2])
                //                               *(FastMath.log(Numerics.max(1.0,vol[obj1][obj2])-logVolMean2[obj1][obj2])
                //                               /Numerics.max(1.0,(logVolStdv2[obj1][obj2]*logVolStdv2[obj1][obj2])));
                //double pdiff = 1.0-FastMath.exp(-0.5*(score-prev[obj])*(score-prev[obj])/(scale*scale));
                //double pcert = FastMath.exp(-0.5*(score-avgbound[obj])*(score-avgbound[obj])/devbound[obj]);
                double pcert = FastMath.exp(-0.5*(score*score)/devdiff[obj1][obj2]);
                
                double pstop = pvol*pcert;
                
                if (pstop>bestproba[obj1] && vol[obj1][obj2]>FastMath.exp(logVolMean2[obj1][obj2]-spread*logVolStdv2[obj1][obj2])) {
                    bestproba[obj1] = pstop;
                    bestvol[obj1] = 0.0f;
                    for (int obj=0;obj<nobj;obj++) bestvol[obj1] += vol[obj1][obj];
                }
                // update the values
                
                // run until the volume exceeds the mean volume + n*stdev
                if (vol[obj1][obj2]<FastMath.exp(logVolMean2[obj1][obj2]+spread*logVolStdv2[obj1][obj2])) {
                    // add neighbors
                    //for (byte k = 0; k<6; k++) {
                    for (byte k = 0; k<26; k++) {
                        int ngb = Ngb.neighborIndex(k, xyz, nx, ny, nz);
                        if (ngb>0 && ngb<nxyz && idmap[ngb]>-1) {
                            if (mask[ngb]) {
                                if (labels[idmap[ngb]]==0) {
                                    for (int best=0;best<nbest;best++) {
                                        // same label
                                        if (combinedLabels[best][idmap[ngb]]==obj1obj2) {
                                            if (best==0) {
                                                heap.addValue(combinedProbas[0][idmap[ngb]]-combinedProbas[1][idmap[ngb]],ngb,obj1obj2);
                                            } else {
                                                heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[0][idmap[ngb]],ngb,obj1obj2);
                                            }
                                            best=nbest;
                                        // boundary labels    
                                        } else if (combinedLabels[best][idmap[ngb]]>100*(obj1+1) && combinedLabels[best][idmap[ngb]]<100*(obj1+2)) {
                                            if (best==0) {
                                                heap.addValue(combinedProbas[0][idmap[ngb]]-combinedProbas[1][idmap[ngb]],ngb,combinedLabels[best][idmap[ngb]]);
                                            } else {
                                                heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[0][idmap[ngb]],ngb,combinedLabels[best][idmap[ngb]]);
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
        }
        System.out.println("\nOptimized volumes: ");
        //for (int obj1=1;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) System.out.println(obj1+" | "+obj2+": "+bestvol[obj1][obj2]+" ("+bestproba[obj1][obj2]+") ");
        for (int obj=1;obj<nobj;obj++) System.out.println(obj+": "+bestvol[obj]+" ("+bestproba[obj]+") ");
        // re-run one last time to get the segmentation
        heap.reset();
        for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
            vol[obj1][obj2] = 0.0;
        }
        for(int id=0;id<ndata;id++) labels[id] = 0;
        for (int obj1=1;obj1<nobj;obj1++) {
            heap.addValue(bestscore[obj1],start[obj1],100*(obj1+1)+(obj1+1));
        }
        while (heap.isNotEmpty()) {
            float score = heap.getFirst();
            int xyz = heap.getFirstId1();
            int obj1obj2 = heap.getFirstId2();
            heap.removeFirst();
            if (labels[idmap[xyz]]==0) {
                int obj1 = Numerics.floor(obj1obj2/100)-1;
                int obj2 = obj1obj2 - 100*(obj1+1) - 1;
                double volsum = 0.0;
                for (int obj=0;obj<nobj;obj++) volsum += vol[obj1][obj];
                if (volsum<bestvol[obj1]) {
                    // update the values
                    vol[obj1][obj2]+=rx*ry*rz;
                    labels[idmap[xyz]] = obj1obj2;
                
                    // add neighbors
                    //for (byte k = 0; k<6; k++) {
                    for (byte k = 0; k<26; k++) {
                        int ngb = Ngb.neighborIndex(k, xyz, nx, ny, nz);
                        if (ngb>0 && ngb<nxyz && idmap[ngb]>-1) {
                            if (mask[ngb]) {
                                if (labels[idmap[ngb]]==0) {
                                    for (int best=0;best<nbest;best++) {
                                        // same label
                                        if (combinedLabels[best][idmap[ngb]]==obj1obj2) {
                                            if (best==0) {
                                                heap.addValue(combinedProbas[0][idmap[ngb]]-combinedProbas[1][idmap[ngb]],ngb,obj1obj2);
                                            } else {
                                                heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[0][idmap[ngb]],ngb,obj1obj2);
                                            }
                                            best=nbest;
                                        // boundary labels    
                                        } else if (combinedLabels[best][idmap[ngb]]>100*(obj1+1) && combinedLabels[best][idmap[ngb]]<100*(obj1+2)) {
                                            if (best==0) {
                                                heap.addValue(combinedProbas[0][idmap[ngb]]-combinedProbas[1][idmap[ngb]],ngb,combinedLabels[best][idmap[ngb]]);
                                            } else {
                                                heap.addValue(combinedProbas[best][idmap[ngb]]-combinedProbas[0][idmap[ngb]],ngb,combinedLabels[best][idmap[ngb]]);
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
        }
        // final segmentation: collapse onto result images
        finalLabel = new int[nxyz];
        finalProba = new float[nxyz];
        float[] kept = new float[nobj];
        for (int x=1;x<ntx-1;x++) for (int y=1;y<nty-1;y++) for (int z=1;z<ntz-1;z++) {
            int xyz = x+ntx*y+ntx*nty*z;
            if (mask[xyz]) {
                if (labels[idmap[xyz]]==0) labels[idmap[xyz]] = 101;
                int obj = Numerics.floor(labels[idmap[xyz]]/100)-1;
                for (int best=0;best<nbest;best++) {
                    if (combinedLabels[best][idmap[xyz]]==labels[idmap[xyz]]) {
                    //if (combinedLabels[best][idmap[xyz]]>100*(obj+1) && combinedLabels[best][idmap[xyz]]<100*(obj+2)) {
                        finalProba[xyz] = combinedProbas[best][idmap[xyz]];
                        best=nbest;
                    }
                }
                finalLabel[xyz] = obj;
            }
        }
        return;            
	}
		
}

