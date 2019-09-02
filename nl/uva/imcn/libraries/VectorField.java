package nl.uva.imcn.libraries;

import java.io.*;
import java.util.*;
import nl.uva.imcn.utilities.*;

/**
 *
 *  This class computes various basic vector field functions.
 *	
 *	@version    Feb 2011
 *	@author     Pierre-Louis Bazin
 *		
 *
 */

public class VectorField {
	
	// no data: used as a library of functions
	
    // simple image functions
    
    public static final byte X = 0;
    public static final byte Y = 1;
    public static final byte Z = 2;
    public static final byte T = 3;

	/**
	*	vector calculus: div (assuming isotropic voxels)
	 */
	public static float[] divergence3D(float[][] vect, int nx, int ny, int nz) {
		float[] div;
		div = new float[nx*ny*nz];
			
        for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
        	int xyz = x+nx*y+nx*ny*z;
			div[xyz] = 0.5f*(vect[X][xyz+1]-vect[X][xyz-1] 
							+vect[X][xyz+nx]-vect[X][xyz-nx] 
							+vect[X][xyz+nx*ny]-vect[X][xyz-nx*ny]);
		}
		return div;
	}
	
	/**
	*	vector calculus: curl (assuming isotropic voxels)
	 */
	public static float[][] curl3D(float[][] vect, int nx, int ny, int nz) {
		float[][] curl;
		curl = new float[3][nx*ny*nz];
			
        for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
        	int xyz = x+nx*y+nx*ny*z;
			curl[X][xyz] = 0.5f*(vect[Y][xyz+nx*ny]-vect[Y][xyz-nx*ny] - vect[Z][xyz+   nx]-vect[Z][xyz-   nx]);
			curl[Y][xyz] = 0.5f*(vect[Z][xyz+    1]-vect[Z][xyz-    1] - vect[X][xyz+nx*ny]-vect[X][xyz-nx*ny]);
			curl[Z][xyz] = 0.5f*(vect[X][xyz+   nx]-vect[X][xyz-   nx] - vect[Y][xyz+    1]-vect[Y][xyz-    1]);
		}
		return curl;
	}
		
	/**
	 *	vector calculus : return the divergence-free part of the vector
	 *	adapted from J.Stam's fast solver (assuming isotropic voxels)
	 *  (note: additional boundary conditions are needed if the field is not zero at the image boundaries)
	 */
	public static final float[][] rotational3D(float[][] vect, int nx, int ny, int nz) {
		return rotational3D(vect, nx, ny, nz, 20);
	}
	public static final float[][] rotational3D(float[][] vect, int nx, int ny, int nz, int niter) {
		float[] div, p;
		float[][] rot;
		div = new float[nx*ny*nz];
		p = new float[nx*ny*nz];
		rot = new float[3][nx*ny*nz];
		
		for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
        	int xyz = x+nx*y+nx*ny*z;
			div[xyz] = -0.5f*(vect[X][xyz+1]		-vect[X][xyz-1] 
						 	 +vect[Y][xyz+nx]	-vect[Y][xyz-nx] 
							 +vect[Z][xyz+nx*ny]	-vect[Z][xyz-nx*ny]);
			p[xyz] = 0.0f;
		}
			
		for (int k=0;k<niter;k++) {
			for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
				int xyz = x+nx*y+nx*ny*z;
				p[xyz] = (div[xyz] + p[xyz-1]+p[xyz+1]+p[xyz-nx]+p[xyz+nx]+p[xyz-nx*ny]+p[xyz+nx*ny])/6;
			}
		}

		for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
        	int xyz = x+nx*y+nx*ny*z;
        	rot[X][xyz] = vect[X][xyz] - 0.5f*(p[xyz+1]-p[xyz-1]);
        	rot[Y][xyz] = vect[Y][xyz] - 0.5f*(p[xyz+nx]-p[xyz-nx]);
        	rot[Z][xyz] = vect[Z][xyz] - 0.5f*(p[xyz+nx*ny]-p[xyz-nx*ny]);
        }
		return rot;
	}
	
	/**
	 *	vector calculus : return the curl-free part of the vector
	 *	adapted from J.Stam's fast solver (assuming isotropic voxels)
	 */
	public static float[][] solenoidal3D(float[][] vect, int nx, int ny, int nz) {
		float[] div, p;
		float[][] sol;
		div = new float[nx*ny*nz];
		p = new float[nx*ny*nz];
		sol = new float[3][nx*ny*nz];
			
        for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
        	int xyz = x+nx*y+nx*ny*z;
			div[xyz] = -0.5f*(vect[X][xyz+1]-vect[X][xyz-1] 
							 +vect[X][xyz+nx]-vect[X][xyz-nx] 
							 +vect[X][xyz+nx*ny]-vect[X][xyz-nx*ny]);
			p[xyz] = 0.0f;
		}
		
		for (int k=0;k<20;k++) {
			for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
				int xyz = x+nx*y+nx*ny*z;
				p[xyz] = (div[xyz] + p[xyz-1]+p[xyz+1]+p[xyz-nx]+p[xyz+nx]+p[xyz-nx*ny]+p[xyz+nx*ny])/6;
			}
		}

		for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
        	int xyz = x+nx*y+nx*ny*z;
        	sol[X][xyz] = 0.5f*(p[xyz+1]-p[xyz-1]);
        	sol[Y][xyz] = 0.5f*(p[xyz+nx]-p[xyz-nx]);
        	sol[Z][xyz] = 0.5f*(p[xyz+nx*ny]-p[xyz-nx*ny]);
        }
		return sol;
	}
	
	/** simplified divergence filter (from rotational3D) */
	public static final float[][] reduceDivergence(float[][] vect, int nx, int ny, int nz, float factor) {
		float[] div;
		float[][] rot;
		div = new float[nx*ny*nz];
		rot = new float[3][nx*ny*nz];
			
        for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
        	int xyz = x+nx*y+nx*ny*z;
			div[xyz] = -0.5f*(vect[X][xyz+1]		-vect[X][xyz-1] 
							 +vect[Y][xyz+nx]	-vect[Y][xyz-nx] 
							 +vect[Z][xyz+nx*ny]	-vect[Z][xyz-nx*ny]);
		}

		for (int x=1;x<nx-1;x++) for (int y=1;y<ny-1;y++) for (int z=1;z<nz-1;z++) {
        	int xyz = x+nx*y+nx*ny*z;
        	rot[X][xyz] = vect[X][xyz] - 0.5f*factor*(div[xyz+1]-div[xyz-1]);
        	rot[Y][xyz] = vect[Y][xyz] - 0.5f*factor*(div[xyz+nx]-div[xyz-nx]);
        	rot[Z][xyz] = vect[Z][xyz] - 0.5f*factor*(div[xyz+nx*ny]-div[xyz-nx*ny]);
        }
		return rot;
	}

}
