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
	
	private int nsub;
	private int nobj;
	private int nc;
	private int nbest = 8;
	
	private float deltaIn = 2.0f;
	private float deltaOut = 0.0f;
	private float boundary = 5.0f;
	private boolean modelBackground = true;
	private boolean cancelBackground = true;
	private boolean cancelAll = false;
	private boolean sumPosterior = false;
	private boolean maxPosterior = false;
	//private boolean topoParam = true;
	//private     String	            lutdir = null;
	
	private float[] 		probaImages;
	private int[] 			labelImages;
	
	private int nx, ny, nz, nxyz;
	private float rx, ry, rz;

	public final void setNumberOfSubjectsObjectsAndContrasts(int sub,int obj,int cnt) {
	    nsub = sub;
	    nobj = obj;
	    nc = cnt;
	    lvlImages = new float[nsub][nobj][];
	    intensImages = new float[nsub][nc][];
	    targetImages = new float[nc][];
	}
	public final void setLevelsetImageAt(int sub, int obj, float[] val) { lvlImages[sub][obj] = val; 
	    System.out.println("levelset ("+sub+", "+obj+") = "+lvlImages[sub][obj][Numerics.floor(nx/2+nx*ny/2+nx*ny*nz/2)]);
	}
	public final void setContrastImageAt(int sub, int cnt, float[] val) { intensImages[sub][cnt] = val; 
	    System.out.println("contrast ("+sub+", "+cnt+") = "+intensImages[sub][cnt][Numerics.floor(nx/2+nx*ny/2+nx*ny*nz/2)]);
	}
	public final void setTargetImageAt(int cnt, float[] val) { targetImages[cnt] = val; 
	    System.out.println("target ("+cnt+") = "+targetImages[cnt][Numerics.floor(nx/2+nx*ny/2+nx*ny*nz/2)]);
	}
	
	public final void setOptions(boolean mB, boolean cB, boolean cA, boolean sP, boolean mP) {
	    modelBackground = mB;
	    cancelBackground = cB;
	    cancelAll = cA;
	    sumPosterior = sP;
	    maxPosterior = mP;
	}
	
	//public static final void setFollowSkeleton(boolean val) { skelParam=val; }
	//public final void setCorrectSkeletonTopology(boolean val) { topoParam=val; }
	//public final void setTopologyLUTdirectory(String val) { lutdir = val; }

	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
	
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
	public final float[] getBestProbabilityMaps() { return probaImages; }
	public final int[] getBestProbabilityLabels() { return labelImages; }

	public void execute() {
	    
	    System.out.println("dimensions: "+nsub+" subjects, "+nc+" contrasts, "+nobj+" objects");
		
	    float[][][] levelsets = null; 
	    
	    // not correct: explicitly build the levelset of the background first, then crop it
	    if (modelBackground) {
            // adding the background: building a ring around the structures of interest
            // with also a sharp decay to the boundary
            levelsets = new float[nsub][nobj+1][];
            float[] background = new float[nxyz];
            boolean[] bgmask = new boolean[nxyz];
            for (int xyz=0;xyz<nxyz;xyz++) bgmask[xyz] = true;
 
            for (int sub=0;sub<nsub;sub++) {
                for (int xyz=0;xyz<nxyz;xyz++) {
                    float mindist = boundary;
                    for (int obj=0;obj<nobj;obj++) {
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
                for (int obj=0;obj<nobj;obj++) {
                    levelsets[sub][obj+1] = lvlImages[sub][obj];
                }
            }
            nobj = nobj+1;
        } else {
            levelsets = lvlImages;
		}
		// mask anything too far outside the structures of interest
		boolean[] mask = new boolean[nxyz];
		int ndata = 0;
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
        System.out.println("masking: compress to "+(ndata/(float)nxyz));
		
        // adapt number of kept values?
        
		System.out.println("compute joint conditional shape priors");
		float[][] probas = new float[nbest][ndata]; 
		int[][] labels = new int[nbest][ndata];
		
		int ctr = Numerics.floor(nsub/2);
        int dev = Numerics.floor(nsub/4);
                    
		double[] val = new double[nsub];
		int id=0;
		double iqrsum=0, iqrden=0;
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    double[][] priors = new double[nobj][nobj];
            for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                // cancelling self-structures?
                if (cancelBackground && obj1==0 && obj2==0) {
                    priors[obj1][obj2] = 0.0;
                } else if (cancelAll && obj1==obj2) {
                    priors[obj1][obj2] = 0.0;
                } else {
                    // median and iqr
                    for (int sub=0;sub<nsub;sub++) {
                        val[sub] = Numerics.max(levelsets[sub][obj1][xyz]-deltaOut, levelsets[sub][obj2][xyz]-deltaIn);
                        val[sub] = Numerics.max(val[sub], 0.0);
                    }
                    // problem when dealing with few samples??
                    /*
                    Percentile measure = new Percentile();
                    measure.setData(val);
                
                    double med = measure.evaluate(50.0); 
                    double iqr = measure.evaluate(75.0) - measure.evaluate(25.0);
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
                    iqrsum += iqr;
                    iqrden++;
                    // IQR = 1.349*sigma
                    // pb: shouldn't we scale by 1/sqrt 2pi sigma ?? 
                    // otherwise more variable regions are preferred
                    
                    // for debug only
                    //med = Numerics.max(0.5*(val[0]+val[1]),0.0);
                    //iqr = Numerics.abs(0.5*(val[0]-val[1]));
                    
                    // arbitrary floor of iqr to 1 voxel
                    //iqr = Numerics.max(iqr, 1.0);
                    
                    // iqr is too variable at the voxel level: keep it constant at first
                    // then re-estimate it at the object|object level
                    iqr = Numerics.max(iqr, 0.5);
                    
                    priors[obj1][obj2] = 1.0/FastMath.sqrt(2.0*FastMath.PI*1.349*iqr*1.349*iqr)*FastMath.exp( -0.5*med*med/(1.349*iqr*1.349*iqr) );
                    //priors[obj1][obj2] = FastMath.exp( -0.5*med*med/(1.349*iqr*1.349*iqr) );
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
                labels[best][id] = 100*(best1+1)+(best2+1);
                probas[best][id] = (float)priors[best1][best2];
                // remove best value
                priors[best1][best2] = 0.0;
 		    }
 		    id++;
		}
		System.out.println("mean spatial iqr: "+(iqrsum/iqrden));
		// levelsets are now discarded...
		levelsets = null;
		
		System.out.println("compute joint conditional intensity priors");
		
		float[][][] contrasts = intensImages;
		float[][] medc = new float[nc][ndata];
		float[][] iqrc = new float[nc][ndata];
		
		System.out.println("1. estimate subjects distribution");
		double[] cntsum = new double[nc];
		double[] cntden = new double[nc];
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
		
		// use spatial priors and subject variability priors to define conditional intensity
		// mean and stdev
		double[][][] condmean = new double[nc][nobj][nobj];
		double[][][] condstdv = new double[nc][nobj][nobj];
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
                           if (labels[best][id]==100*(obj1+1)+(obj2+1)) {
                               // found value: proceeed
                               for (int sub=0;sub<nsub;sub++) {
                                   double psub = probas[best][id]*1.0/FastMath.sqrt(2.0*FastMath.PI*1.349*iqr*1.349*iqr)
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
		       } else {
		           System.out.print("empty pair        ");
		           condmean[c][obj1][obj2] = 0.0;
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
                           if (labels[best][id]==100*(obj1+1)+(obj2+1)) {
                               // found value: proceeed
                               for (int sub=0;sub<nsub;sub++) {
                                   double psub = probas[best][id]*1.0/FastMath.sqrt(2.0*FastMath.PI*1.349*iqr*1.349*iqr)
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
		       } else if (den>0) {
		           condstdv[c][obj1][obj2] = FastMath.sqrt(var/den);
		           System.out.print(condmean[c][obj1][obj2]+" +/- "+condstdv[c][obj1][obj2]+"    ");
		       
                } else {
                   System.out.print("empty pair        ");
		           condstdv[c][obj1][obj2] = 0;
		       } 
		    }
		}
		// at this point the atlas data is not used anymore
		contrasts = null;
		System.out.println("\ndone");
          
        System.out.println("apply priors to target");
        
		// combine priors and contrasts posteriors (update the priors maps)
		float[][] target = targetImages;
		id=0;
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		   double[][] posteriors = new double[nobj][nobj];
		   for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
               // look for non-zero priors
               posteriors[obj1][obj2] = 0.0;
           
               for (int best=0;best<nbest;best++) {
                   if (labels[best][id]==100*(obj1+1)+(obj2+1)) {
                       // multiply nc times to balance prior and posterior
                       posteriors[obj1][obj2] = 1.0;
                       for (int c=0;c<nc;c++) posteriors[obj1][obj2] *= probas[best][id];
                   }
               }
               if (posteriors[obj1][obj2]>0) {
                   for (int c=0;c<nc;c++) {
                       if (condstdv[c][obj1][obj2]>0) {
                           double pobjc = 1.0/FastMath.sqrt(2.0*FastMath.PI*condstdv[c][obj1][obj2]*condstdv[c][obj1][obj2])
                                           *FastMath.exp( -0.5*(target[c][xyz]-condmean[c][obj1][obj2])*(target[c][xyz]-condmean[c][obj1][obj2])
                                                               /(condstdv[c][obj1][obj2]*condstdv[c][obj1][obj2]) );
                           posteriors[obj1][obj2] *= pobjc;
                       } else {
                           // what to do here? does it ever happen?
                           //System.out.print("!");
                       }
                   }
               }
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
                labels[best][id] = 100*(best1+1)+(best2+1);
                probas[best][id] = (float)posteriors[best1][best2];
                // remove best value
                posteriors[best1][best2] = 0.0;
 		    }
 		    id++;
		}
		// rebuild output
		target = null;
		
		probaImages = new float[nbest*nxyz];
		labelImages = new int[nbest*nxyz];
		id=0;
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int best=0;best<nbest;best++) {
                probaImages[xyz+best*nxyz] = probas[best][id];
                labelImages[xyz+best*nxyz] = labels[best][id];
            }
            id++;
        }
		
        return;
	}


}
