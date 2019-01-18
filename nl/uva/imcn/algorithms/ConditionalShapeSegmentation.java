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
	private int nbest = 5;
	
	private float deltaIn = 1.0f;
	private float deltaOut = 0.0f;
	private float boundary = 10.0f;
	private boolean modelBackground = true;
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
	public final void setLevelsetImageAt(int sub, int obj, float[] val) { lvlImages[sub][obj] = val; }
	public final void setContrastImageAt(int sub, int cnt, float[] val) { intensImages[sub][cnt] = val; }
	public final void setTargetImageAt(int cnt, float[] val) { targetImages[cnt] = val; }
	
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
            levelsets = new float[nsub][nobj+1][nxyz];
            for (int sub=0;sub<nsub;sub++) for (int xyz=0;xyz<nxyz;xyz++) {
                float mindist = boundary;
                for (int obj=0;obj<nobj;obj++) {
                    if (lvlImages[sub][obj][xyz]<mindist) mindist = lvlImages[sub][obj][xyz];
                }
                if (mindist<boundary/2.0) {
                    levelsets[sub][0][xyz] = -mindist;
                } else {
                   levelsets[sub][0][xyz] = -boundary/2.0f + 3.0f*(mindist-boundary/2.0f);
                }
                for (int obj=0;obj<nobj;obj++) {
                    levelsets[sub][obj+1][xyz] = lvlImages[sub][obj][xyz];
                }
            }
            nobj = nobj+1;
            lvlImages = null;
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
		
		System.out.println("compute joint conditional shape priors");
		float[][] probas = new float[nbest][ndata]; 
		int[][] labels = new int[nbest][ndata];
		
		double[] val = new double[nsub];
		int id=0;
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    double[][] priors = new double[nobj][nobj];
            for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
                // median and iqr
                for (int sub=0;sub<nsub;sub++) {
                    val[sub] = Numerics.max(levelsets[sub][obj1][xyz]-deltaOut, levelsets[sub][obj2][xyz]-deltaIn);
		        }
		        Percentile measure = new Percentile();
                measure.setData(val);
			
                double med = Numerics.max(measure.evaluate(50.0), 0.0); 
                double iqr = measure.evaluate(75.0) - measure.evaluate(25.0);
				// IQR = 1.349*sigma
				// pb: shouldn't we scale by 1/sqrt 2pi sigma ?? 
				// otherwise more variable regions are preferred
				
				// arbitrary floor of iqr to 1 voxel
				iqr = Numerics.max(iqr, 1.0);
				
				// for debug only
				//med = Numerics.max(0.5*(val[0]+val[1]),0.0);
				//iqr = Numerics.abs(0.5*(val[0]-val[1]));
				//iqr = 1.0;
				
				priors[obj1][obj2] = 1.0/FastMath.sqrt(2.0*FastMath.PI*1.349*iqr*1.349*iqr)*FastMath.exp( -0.5*med*med/(1.349*iqr*1.349*iqr) );
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
		// levelsets are now discarded...
		levelsets = null;
		
		/* skip for now
		System.out.println("compute joint conditional intensity priors");
		
		float[][][] contrasts = intensImages;
		float[][] medc = new float[nc][ndata];
		float[][] iqrc = new float[nc][ndata];
		
		System.out.println("1. estimate subjects distribution");
		id = 0;
		for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		    for (int c=0;c<nc;c++) {
                for  (int sub=0;sub<nsub;sub++) {
                    val[sub] = contrasts[sub][c][xyz];
                }
                //System.out.println("values");
                Percentile measure = new Percentile();
                measure.setData(val);
			
                medc[c][id] = (float)measure.evaluate(50.0); 
                //System.out.println("median "+medc[c][id]);
                iqrc[c][id] = (float)(measure.evaluate(75.0) - measure.evaluate(25.0));
                //System.out.println("iqr "+iqrc[c][id]);
            }
            id++;
        }
        
        System.out.println("2. compute conditional maps");
		
		// use spatial priors and subject variability priors to define conditional intensity
		// mean and stdev
		double[][][] condmean = new double[nc][nobj][nobj];
		double[][][] condstdv = new double[nc][nobj][nobj];
		for (int c=0;c<nc;c++) {
		   for (int obj1=0;obj1<nobj;obj1++) for (int obj2=0;obj2<nobj;obj2++) {
		       
		       System.out.print("("+obj1+" | "+obj2+")");
		       // System.out.println("..mean");
		       double sum = 0.0;
		       double den = 0.0;
		       id = 0;
		       for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		           // look for non-zero priors
		           for (int best=0;best<nbest;best++) {
		               if (labels[best][id]==100*(obj1+1)+(obj2+1)) {
		                   // found value: proceeed
		                   for (int sub=0;sub<nsub;sub++) {
		                       double med = medc[c][id];
		                       double iqr = iqrc[c][id];
		                       double psub = 0.0;
		                       // assuming here that iqr==0 means masked regions
		                       if (iqr>0) psub = probas[best][id]*1.0/FastMath.sqrt(2.0*FastMath.PI*1.349*iqr*1.349*iqr)
		                                          *FastMath.exp( -0.5*(contrasts[sub][c][xyz]-med)*(contrasts[sub][c][xyz]-med)/(1.349*iqr*1.349*iqr) );
		                       // add to the mean
		                       sum += psub*contrasts[sub][c][xyz];
		                       den += psub;
		                   }
		                   best=nbest;
		               }
		           }
		           id++;
		       }
		       // build average
		       condmean[c][obj1][obj2] = sum/den;
		       
		       //System.out.println("..stdev");
		       double var = 0.0;
		       id = 0;
		       for (int xyz=0;xyz<nxyz;xyz++) if (mask[xyz]) {
		           // look for non-zero priors
		           for (int best=0;best<nbest;best++) {
		               if (labels[best][id]==100*(obj1+1)+(obj2+1)) {
		                   // found value: proceeed
		                   for (int sub=0;sub<nsub;sub++) {
		                       double med = medc[c][id];
		                       double iqr = iqrc[c][id];
		                       double psub = 0.0;
		                        // assuming here that iqr==0 means masked regions
		                       if (iqr>0) psub = probas[best][id]*1.0/FastMath.sqrt(2.0*FastMath.PI*1.349*iqr*1.349*iqr)
		                                           *FastMath.exp( -0.5*(contrasts[sub][c][xyz]-med)*(contrasts[sub][c][xyz]-med)/(1.349*iqr*1.349*iqr) );
		                       // add to the mean
		                       var += psub*(contrasts[sub][c][xyz]-condmean[c][obj1][obj2])*(contrasts[sub][c][xyz]-condmean[c][obj1][obj2]);
		                   }
		                   best=nbest;
		               }
		           }
		           id++;
		       }
		       // build stdev
		       condstdv[c][obj1][obj2] = FastMath.sqrt(var/den);
		       if (var==0) System.out.println("empty region");
		   }
		}
		// at this point the atlas data is not used anymore
		contrasts = null;
		System.out.println("done");
          
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
                       posteriors[obj1][obj2] = probas[best][id];
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
                           System.out.print("!");
                       }
                   }
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
		*/
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
