// Theo Walker
// Max Planck Florida Institute
// naroom@gmail.com

package cellMagicWand;

import java.util.ArrayList;

import ij.ImagePlus;
import ij.io.FileSaver;
import ij.io.Opener;
import ij.process.ShortProcessor;

public class PolarTest {

	public static void test(){
		String testDataDir = "C:/netbeans-projects/Polar_Cell/src/testData/";
		
		String imagePath = testDataDir + "testCircles.tif";
		//String imagePath = testDataDir + "white.tif";
        Opener o = new Opener();
        ImagePlus img = o.openImage(imagePath);
        
        int[][] circleCenters = new int[][]{
        		{322,217},
        		{268,276},
        		{185,232},
        		{225,360}
        };

        int magFactor = 10;
        ShortProcessor sp = new ShortProcessor(img.getWidth()*magFactor, img.getHeight()*magFactor);
        ShortProcessor sp2 = new ShortProcessor(img.getWidth(), img.getHeight());
        
        //draw a pixel "grid" on the shortProcessor
        for(int x=0; x < img.getWidth()*magFactor; x+=magFactor){
            for(int y=0; y < img.getHeight()*magFactor; y+=magFactor){
				for(int xi=x; xi < x+magFactor; xi++){
	            	sp.set(xi, y+1, 10000);
            	}
				for(int yi=y; yi < y+magFactor; yi++){
	            	sp.set(x+1, yi, 10000);
				}

            }
        }
        
        for(int i = 0; i < circleCenters.length; i++){
        	print("running point " + circleCenters[i][0] + ", " + circleCenters[i][1]);
            final long startTime = System.currentTimeMillis();

        	PolarTransform pt = new PolarTransform(img, circleCenters[i][0], circleCenters[i][1], 
        			2, 15, 0.25, 120, true);
        	
            //write out polar image
            ImagePlus polarImg = pt.getPolarImage();
            FileSaver fs = new FileSaver(polarImg);
            fs.saveAsTiff(testDataDir + "polarImg" +  i+ ".tif");
           
        	
            final long endTime = System.currentTimeMillis();
            print("Time elapsed: " + (endTime-startTime) + "ms");

            
            //draw the edge points
            /*
            int[][] edgePoints = pt.getPolarEdgePoints();
            for(int j = 0; j < edgePoints[0].length; j++){
        		sp2.set(edgePoints[0][j], edgePoints[1][j], 65535);
            }
            */
            
            
            /*
        	ArrayList<RingPixel> ringPixels = pt.getEdgePointsRoi();
        	for(int j = 0; j < ringPixels.size(); j++){
        		int x = ringPixels.get(j).x;
        		int y = ringPixels.get(j).y;
        		int angleRange = (int) Math.round((ringPixels.get(j).thetaMax-ringPixels.get(j).thetaMin)*20000);
                sp2.set(x, y, 65535);
        	}*/
            
            
            /*
        	
        	//draw the magnified edge points
            int[][] edgePointsMag = pt.getEdgePointsMagnified(magFactor);
            for(int j = 0; j < edgePointsMag[0].length; j++){
        		sp.set(edgePointsMag[0][j]+1, edgePointsMag[1][j]+1, 65535);
            }

            //draw the 4-connected edge points *crosses fingers*
            ArrayList<Pixel> cellEdge = pt.getPixelCellEdge();
            print("lol cell edge size is "+ cellEdge.size());
            for(int j = 0; j < cellEdge.size(); j++){
        		sp2.set(cellEdge.get(j).x+1, cellEdge.get(j).y+1, 65535);
            }
            */
        	
        }

        //write out edge image
        ImagePlus magImg = new ImagePlus("",sp);
        FileSaver fs = new FileSaver(magImg);
        fs.saveAsTiff(testDataDir + "magImg"+".tif");
        
        ImagePlus edgeImg = new ImagePlus("",sp2);
        fs = new FileSaver(edgeImg);
        fs.saveAsTiff(testDataDir + "edgeImg"+".tif");

        print("yay");
	}

	public static void main(String[] args){
		test();
	}

	public static void print(String str){
		System.out.println(str);
	}
}
