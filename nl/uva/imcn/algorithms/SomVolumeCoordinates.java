package nl.uva.imcn.algorithms;

import nl.uva.imcn.utilities.*;
import nl.uva.imcn.structures.*;
import nl.uva.imcn.libraries.*;
import nl.uva.imcn.methods.*;

/*
 * @author Pierre-Louis Bazin
 */
public class SomVolumeCoordinates {

	// jist containers
	private float[] 	probaImage;
	private static	int 		nx,ny,nz, nxyz;   		// images dimensions
	private static	float 		rx,ry,rz;   		// images resolutions
	
	private float[] 	mappedSomPoints;
	private int[] 		mappedSomTriangles;
	private float[] 	mappedSomValues;
	
	private float[] 	mappedImage;
	
	private boolean maskZeros = false;
	
	// som parameters
	private int somDim = 2;
	private int somSize = 100;
	private int learningTime = 1000;
	private int totalTime = 5000;
	
	// create inputs
	public final void setProbaImage(float[] val) { probaImage = val; }
	
	public final void setSomDimension(int val) { somDim = val; }
	public final void setSomSize(int val) { somSize = val; }
	public final void setLearningTime(int val) { learningTime = val; }
	public final void setTotalTime(int val) { totalTime = val; }
	
	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
		
	// to be used for JIST definitions, generic info / help
	public final String getPackage() { return "IMCN Toolkit"; }
	public final String getCategory() { return "Dimensionality reduction"; }
	public final String getLabel() { return "Som Volume Coordinates"; }
	public final String getName() { return "SomVolumeCoordinates"; }

	public final String[] getAlgorithmAuthors() { return new String[]{"Pierre-Louis Bazin"}; }
	public final String getAffiliation() { return "Integrative Model-based Cognitive Neuroscience research unit, Universiteit van Amsterdam"; }
	public final String getDescription() { return "Map probability densities in volume space to an ND self-organizing map."; }
	public final String getLongDescription() { return getDescription(); }
		
	public final String getVersion() { return "1.0"; };
			
	// create outputs
	public final float[] 	getMappedSomPoints() { return mappedSomPoints; }
	public final int[] 		getMappedSomTriangles() { return mappedSomTriangles; }
	public final float[] 	getMappedSomValues() { return mappedSomValues; }
	
	public final float[] 	getMappedImage() { return mappedImage; }
	
	public void execute() {

	    // extract the image values
	    // (and rescale them in [0,1])
	    float pmax = 0;
	    int npt = 0;
	    for (int xyz=0;xyz<nxyz;xyz++) if (probaImage[xyz]>0) {
	        if (probaImage[xyz]>pmax) pmax = probaImage[xyz];
	        npt++;
	    }
	    
	    // reformat the data points and probas
	    float[][] data = new float[npt][3];
	    float[] proba = new float[npt];
	    int pt=0;
	    for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
	        int xyz = x+nx*y+nx*ny*z;
	        if (probaImage[xyz]>0) {
                data[pt][0] = x;
                data[pt][1] = y;
                data[pt][2] = z;
                proba[pt] = probaImage[xyz];
                pt++;
            }
        }
	    
	    BasicSom algorithm = new BasicSom(data, proba, null, npt, 3, somDim, somSize, learningTime, totalTime);
	    algorithm.run_som2D();
		
	    System.out.println("preparing output");
		// ouptput: map som onto volume
		float[][] mapping = algorithm.mapSomOnData2D();
		mappedImage = new float[2*nxyz];
		pt=0;
	    for (int x=0;x<nx;x++) for (int y=0;y<ny;y++) for (int z=0;z<nz;z++) {
	        int xyz = x+nx*y+nx*ny*z;
	        if (probaImage[xyz]>0) {
                mappedImage[xyz] = mapping[pt][0];
                mappedImage[xyz+nxyz] = mapping[pt][1];
                pt++;
            }
		}
		
		System.out.println("som output");
		// output: warp som grid onto surface space
		float[][] som = algorithm.getSomWeights();
		int nx = somSize;
		int ny = somSize;

		System.out.println("som points: "+nx*ny);
		mappedSomPoints = new float[nx*ny*3];
		for (int xy=0;xy<nx*ny;xy++) {
		     mappedSomPoints[0+3*xy] = som[0][xy];
		     mappedSomPoints[1+3*xy] = som[1][xy];
		     mappedSomPoints[2+3*xy] = som[2][xy];
		}
		
		int ntriangles = (nx-1)*(ny-1)*2;
		System.out.println("som triangles: "+ntriangles);
		mappedSomTriangles = new int[3*ntriangles];
		int tr=0;
		for (int x=0;x<nx-1;x++) {
		     for (int y=0;y<ny-1;y++) {
		         // top triangle
		         mappedSomTriangles[0+tr] = x+nx*y;
		         mappedSomTriangles[1+tr] = x+1+nx*y;
		         mappedSomTriangles[2+tr] = x+nx*(y+1);
		         tr+=3;
		         // bottom triangle
		         mappedSomTriangles[0+tr] = x+1+nx*y;
		         mappedSomTriangles[1+tr] = x+nx*(y+1);
		         mappedSomTriangles[2+tr] = x+1+nx*(y+1);
		         tr+=3;
		     }
		}
		System.out.println("som values: "+nx*ny);
		mappedSomValues = new float[2*nx*ny];
		for (int x=0;x<nx;x++) {
		    for (int y=0;y<ny;y++) {
		        int xy = x+nx*y;
                mappedSomValues[xy+0*nx*ny] = x;
                mappedSomValues[xy+1*nx*ny] = y;
            }
		}
		System.out.println("done");
		
		return;
	}

}
