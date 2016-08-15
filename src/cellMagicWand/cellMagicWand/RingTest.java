// Theo Walker
// Max Planck Florida Institute
// naroom@gmail.com

package cellMagicWand;


import ij.ImagePlus;
import ij.io.FileSaver;
import ij.io.Opener;
import ij.process.ShortProcessor;

public class RingTest {

	public static void test(){
		String testDataDir = "C:/netbeans-projects/Polar_Cell/src/testData/";
		
		String imagePath = testDataDir + "testCircles.tif";
        Opener o = new Opener();
        ImagePlus img = o.openImage(imagePath);
        
        int[][] circleCenters = new int[][]{
        		{322,217},
        		{268,276},
        		{185,232},
        		{225,360}
        };
        
        
        long startTime = System.currentTimeMillis();
        for(int i = 0; i < circleCenters.length; i++){
            ShortProcessor sp = new ShortProcessor(img.getWidth(), img.getHeight());
        	
        	for(double radius = 2; radius < 150; radius=radius+4){
	            Ring r = new Ring(img, circleCenters[i][0], circleCenters[i][1], radius, true);
	            
	            double thetaIntervalSum = 0;
	            //Note that we don't output the last pixel, as it is a repeat of the first one.
	            for(int j = 0; j < r.ringPixels.length-1; j++){ 
	            	double thetaInterval = r.ringPixels[j].thetaMax - r.ringPixels[j].thetaMin;
	            	thetaIntervalSum += thetaInterval;
	            	int x = r.ringPixels[j].x;
	            	int y = r.ringPixels[j].y;
	            	int intensity = (int)Math.round(thetaInterval*20000 + sp.get(x, y));
	            	//print("" + j + "\t" + thetaInterval + "\t" + intensity);
	            	sp.set(x, y, intensity);
	            }
            	//print("" + "\t" + thetaIntervalSum);
        	}
            
        	/*        	
        	print("running point " + circleCenters[i][0] + ", " + circleCenters[i][1]);
        	PolarTransform pt = new PolarTransform(img, circleCenters[i][0], circleCenters[i][1], 3, 15, 20, true);
            ImagePlus polarImg = pt.getPolarImage();
            FileSaver fs = new FileSaver(polarImg);
            fs.saveAsTiff(testDataDir + "polarImg.tif");
            
            //write out edge image
            int[][] edgePoints = pt.getEdgePoints();
            print("Edge points: "+edgePoints.length);
            for(int j = 0; j < edgePoints.length; j++){
                sp.set(edgePoints[j][0], edgePoints[j][1], 65535);
            }
            */
            final long endTime = System.currentTimeMillis();
            print("" + (endTime-startTime));
            startTime = endTime;
            
            ImagePlus edgeImg = new ImagePlus("",sp);
            FileSaver fs = new FileSaver(edgeImg);
            fs.saveAsTiff(testDataDir + "edgeImg" + (i+1) + ".tif");
        }

        print("yay");
	}


	public static void main(String[] args){
		test();
	}
	
	public static void print(String str){
		System.out.println(str);
	}
}
