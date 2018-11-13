package nl.uva.imcn.algorithms;

import java.io.*;
import java.util.*;

import nl.uva.imcn.structures.*;
import nl.uva.imcn.libraries.*;
import nl.uva.imcn.utilities.*;
import nl.uva.imcn.methods.*;

import org.apache.commons.math3.util.FastMath;

/**
 *
 *  This algorithm uses a total variation algorithm to filter images
 *
 *	@version    Oct 2018
 *	@author     Pierre-Louis Bazin 
 *		
 *
 */
 
public class TotalVariationFiltering {
	
	// data and membership buffers
	private 	float[] 		image;  			// original image
	private		boolean[]		mask;				// masking regions not used in computations
	private static	int 		nx,ny,nz, nxyz;   		// images dimensions
	private static	float 		rx,ry,rz;   		// images resolutions
	
	// parameters
	private float[] inputImage;
	private int[] maskImage = null;
	private 	float 		lambdaScale = 0.05f;		// scaling parameter
	private 	float 		tauStep = 0.125f;		// internal step parameter (default 1/4)
	private 	float 		maxDist = 0.0001f;		// maximum error for stopping
	private 	int 		maxIter = 500;		// maximum number of iterations
    
	private float[] filterImage;
	private float[] residualImage;
    
	// for debug and display
	private static final boolean		debug=true;
	private static final boolean		verbose=true;
	

	public final void setImage(float[] val) { inputImage = val; }
	public final void setMaskImage(int[] val) { maskImage = val; }
	
	public final void setLambdaScale(float val) { lambdaScale = val; }
	public final void setTauStep(float val) { tauStep = val; }
	public final void setMaxDist(float val) { maxDist = val; }
	public final void setMaxIter(int val) { maxIter = val; }
	
	public final void setDimensions(int x, int y, int z) { nx=x; ny=y; nz=z; nxyz=nx*ny*nz; }
	public final void setDimensions(int[] dim) { nx=dim[0]; ny=dim[1]; nz=dim[2]; nxyz=nx*ny*nz; }
	
	public final void setResolutions(float x, float y, float z) { rx=x; ry=y; rz=z; }
	public final void setResolutions(float[] res) { rx=res[0]; ry=res[1]; rz=res[2]; }
		
	// to be used for JIST definitions, generic info / help
	public final String getPackage() { return "IMCN Toolkit"; }
	public final String getCategory() { return "Filtering"; }
	public final String getLabel() { return "Total Variation"; }
	public final String getName() { return "TotalVariation"; }

	public final String[] getAlgorithmAuthors() { return new String[]{"Pierre-Louis Bazin"}; }
	public final String getAffiliation() { return "Integrated Model-based Cognitive Neuroscience Reseaerch Unit, Universiteit van Amsterdam"; }
	public final String getDescription() { return "Total variation filtering algorithm adapted from (Chambolle, 2004)"; }
	public final String getLongDescription() { return getDescription(); }
		
	public final String getVersion() { return "1.0"; };

	// create outputs
	public final float[] getFilteredImage() { return filterImage; }
    public final float[] getResidualImage() { return residualImage; }
    
	public final void execute() {
	    
        mask = new boolean[nx*ny*nz];
        if (maskImage==null) {
            // no mask
            for (int xyz=0;xyz<nxyz;xyz++) mask[xyz] = true;
        } else {
            for (int xyz=0;xyz<nxyz;xyz++) if (maskImage[xyz]>0) mask[xyz] = true;
            maskImage = null;
        }
        image = inputImage;
        
		if (debug) System.out.print("initialization\n");
		
        TotalVariation1D algo = new TotalVariation1D(inputImage,null,nx,ny,nz, lambdaScale, tauStep, maxDist, maxIter);
        algo.solve();
        
        filterImage = algo.exportResult();
        residualImage = algo.exportResidual();

		return;
    }
    
}

