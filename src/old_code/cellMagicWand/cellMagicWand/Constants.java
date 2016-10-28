// Theo Walker
// Max Planck Florida Institute
// naroom@gmail.com

package cellMagicWand;

public class Constants {
	public static double CIRCLE_THETA_MAX = 360; //

	public static final String PROGRAM_NAME = "CellMagicWand";

	public static final int THETA_SAMPLES_FOR_RADIUS_FINDING = 100;

	//Parameters and their default values.
	public static final int DEFAULT_MIN_DIAMETER = 8; //most people image so their cells are at least 10px across
	public static final int DEFAULT_MAX_DIAMETER = 300; //should cover a lot of cases, without being terribly slow
	public static final double DEFAULT_CIRCUMFERENCE_SAMPLE_RATE = 5.0; 
	public static final String BRIGHT_CELLS = "Bright cells on a dark background";
	public static final String DARK_CELLS = "Dark cells on a bright background";
	
	//The sample rate in R determines how finely we chop up the Cartesian image to make the polar image.
	//0.5 and lower produces reliable edges; it's jaggy at 1.
	public static final double PIXELS_PER_R_SAMPLE = 0.5; 
	
	//Padding the polar image gives better dynamic programming results (Sun & Pallottino 2003).
	public static final double POLAR_PADDING_PERCENT = 20; //Set at 0 to 100
	
	/* Probably unused */
    public static final int SHORT_WHITE = 65535; //used in range scaling USHORT type images
    public static final int BYTE_WHITE = 255; //used in range scaling color & byte images
    public static final int GREY_12BIT = 2048;
}
