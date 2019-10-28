package nl.uva.imcn.algorithms;

import nl.uva.imcn.utilities.*;
import nl.uva.imcn.structures.*;
import nl.uva.imcn.libraries.*;
import nl.uva.imcn.methods.*;

/*
 * @author Pierre-Louis Bazin
 */
public class SomSurfaceCoordinates {

	// jist containers
	private float[] 	surfacePoints;
	private int[] 		surfaceTriangles;
	private float[] 	surfaceValues;
	
	private float[] 	mappedSomPoints;
	private int[] 		mappedSomTriangles;
	private float[] 	mappedSomValues;
	
	private float[] 	mappedSurfacePoints;
	private int[] 		mappedSurfaceTriangles;
	private float[] 	mappedSurfaceValues;
	
	private boolean maskZeros = false;
	
	// som parameters
	private int somDim = 2;
	private int somSize = 100;
	private int learningTime = 1000;
	private int totalTime = 5000;
	
	// create inputs
	public final void setSurfacePoints(float[] val) { surfacePoints = val; }
	public final void setSurfaceTriangles(int[] val) { surfaceTriangles = val; }
	public final void setSurfaceValues(float[] val) { surfaceValues = val; }
	
	public final void setMaskZeroValues(boolean val) { maskZeros = val; }

	public final void setSomDimension(int val) { somDim = val; }
	public final void setSomSize(int val) { somSize = val; }
	public final void setLearningTime(int val) { learningTime = val; }
	public final void setTotalTime(int val) { totalTime = val; }
	
	// to be used for JIST definitions, generic info / help
	public final String getPackage() { return "IMCN Toolkit"; }
	public final String getCategory() { return "Dimensionality reduction"; }
	public final String getLabel() { return "Som Surface Coordinates"; }
	public final String getName() { return "SomSurfaceCoordinates"; }

	public final String[] getAlgorithmAuthors() { return new String[]{"Pierre-Louis Bazin"}; }
	public final String getAffiliation() { return "Integrative Model-based Cognitive Neuroscience research unit, Universiteit van Amsterdam"; }
	public final String getDescription() { return "Map surface coordinates to an ND self-organizing map."; }
	public final String getLongDescription() { return getDescription(); }
		
	public final String getVersion() { return "1.0"; };
			
	// create outputs
	public final float[] 	getMappedSomPoints() { return mappedSomPoints; }
	public final int[] 		getMappedSomTriangles() { return mappedSomTriangles; }
	public final float[] 	getMappedSomValues() { return mappedSomValues; }
	
	public final float[] 	getMappedSurfacePoints() { return mappedSurfacePoints; }
	public final int[] 		getMappedSurfaceTriangles() { return mappedSurfaceTriangles; }
	public final float[] 	getMappedSurfaceValues() { return mappedSurfaceValues; }
	
	public void execute() {

	    // reformat the surface points
	    int npt = surfacePoints.length/3;
	    float[][] data = new float[npt][3];
	    for (int p=0;p<npt;p++) for (int i=0;i<3;i++) data[p][i] = surfacePoints[i + 3*p];
	    
	    boolean[] mask = null;
	    if (maskZeros) {
	        mask = new boolean[npt];
	        for (int p=0;p<npt;p++) {
	            if (surfaceValues[p]==0) mask[p] = false;
                else mask[p] = true;
            }
        }
	    
	    BasicSom algorithm = new BasicSom(data, null, mask, npt, 3, somDim, somSize, learningTime, totalTime);
	    algorithm.run_som2D();
		
	    System.out.println("preparing output");
		// ouptput: map som onto surface
		mappedSurfacePoints = surfacePoints;
		mappedSurfaceTriangles = surfaceTriangles;
		float[][] mapping = algorithm.mapSomOnData2D();
		mappedSurfaceValues = new float[2*npt];
		for (int p=0;p<npt;p++) {
		    mappedSurfaceValues[p] = mapping[p][0];
		    mappedSurfaceValues[p+npt] = mapping[p][1];
		}
		
		System.out.println("som output");
		// output: warp som grid onto surface space
		float[][] som = algorithm.getSomWeights();
		boolean[] lattice = algorithm.getSomShape();
		int nx = somSize;
		int ny = somSize;

		// trim the missing points
		int npts = 0;
		for (int xy=0;xy<nx*ny;xy++) {
		    if (lattice[xy]) npts++;
		}
		
		System.out.println("som points: "+npts);
		mappedSomPoints = new float[npts*3];
		int[] mappedId = new int[nx*ny];
		int id=0;
		for (int xy=0;xy<nx*ny;xy++) if (lattice[xy]) {
		     mappedSomPoints[0+3*id] = som[0][xy];
		     mappedSomPoints[1+3*id] = som[1][xy];
		     mappedSomPoints[2+3*id] = som[2][xy];
		     mappedId[xy] = id;
		     id++;
		}
		
		// trim the missing triangles
		int ntriangles = 0;
		for (int x=0;x<nx-1;x++) {
		     for (int y=0;y<ny-1;y++) { 
		         if (lattice[x+nx*y] && lattice[x+1+nx*y] && lattice[x+nx*(y+1)]) {
		             ntriangles++;
		         }
		         if (lattice[x+1+nx*y] && lattice[x+nx*(y+1)] && lattice[x+1+nx*(y+1)]) {
		             ntriangles++;
		         }
		     }
		}
		System.out.println("som triangles: "+ntriangles);
		mappedSomTriangles = new int[3*ntriangles];
		int tr=0;
		for (int x=0;x<nx-1;x++) {
		     for (int y=0;y<ny-1;y++) {
		         if (lattice[x+nx*y] && lattice[x+1+nx*y] && lattice[x+nx*(y+1)]) {
		             // top triangle
                     mappedSomTriangles[0+tr] = mappedId[x+nx*y];
                     mappedSomTriangles[1+tr] = mappedId[x+1+nx*y];
                     mappedSomTriangles[2+tr] = mappedId[x+nx*(y+1)];
                     tr+=3;
                 }
		         if (lattice[x+1+nx*y] && lattice[x+nx*(y+1)] && lattice[x+1+nx*(y+1)]) {
                     // bottom triangle
                     mappedSomTriangles[0+tr] = mappedId[x+1+nx*y];
                     mappedSomTriangles[1+tr] = mappedId[x+nx*(y+1)];
                     mappedSomTriangles[2+tr] = mappedId[x+1+nx*(y+1)];
                     tr+=3;
                 }
		     }
		}
		System.out.println("som values: "+npts);
		mappedSomValues = new float[2*npts];
		int mp=0;
		for (int x=0;x<nx;x++) {
		    for (int y=0;y<ny;y++) {
		        int xy = x+nx*y;
                if (lattice[xy]) {
                    mappedSomValues[mp+0*npts] = x;
                    mappedSomValues[mp+1*npts] = y;
                    mp++;
                }
            }
		}
		System.out.println("done");
		
		System.out.println("back to python...");
		
		return;
	}

}
