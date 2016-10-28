// Theo Walker
// Max Planck Florida Institute
// naroom@gmail.com

package cellMagicWand;

public class RingPixel {
	// a RingPixel is a single pixel from the original image
	// it is associated with a specific ring (e.g. r=5).
	
	// Note that the mapping from cartesian pixels to ring pixels is not 1:1 -- 
	// some cartesian pixels will be on 2 rings, and hence have 2 ring 
	// pixels associated with them.
	
	//see Ring.java for more details on what this structure is all about.
	
	public int x; //x, y are cartesian coordinates from the original image
	public int y; 
	public double intensity; 
	public double thetaMin; //the lowest value of theta that's still inside this pixel for a given r
	public double thetaMax; 
	public double r; 
	
	public RingPixel(){}

}
