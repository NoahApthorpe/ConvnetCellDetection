// Theo Walker
// Max Planck Florida Institute
// naroom@gmail.com

package cellMagicWand;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

import ij.ImagePlus;
import ij.process.ShortProcessor;


public class PolarTransform {
	Ring[] rings;
	PolarPixel[][] polarImage;
	int paddingSize;
	ArrayList<PolarPixel> polarEdge; //the edge of the cell, in polar pixels. Found by dynamic programming.
	ArrayList<CartesianPixel> cellEdge; //the edge of the cell, in Cartesian pixels. An intermediate used for computation.
	ArrayList<Pixel> pixelCellEdge; //the edge of the cell, in... just regular pixels. This is what gets returned.
	
	//input data
	int cx;
	int cy;
	int rMin;
	int rMax;
	int numThetaSamples;
	double rIncrement;
	int numRSamples;
	boolean cellsAreBright;
	ImagePlus originalImage; 

	//Constructor. Builds the polar image. Calls function to find edge using dynamic programming.
	public PolarTransform(ImagePlus originalImage, int cx, int cy, 
			int rMin, int rMax, double rIncrement, int numThetaSamples, boolean cellsAreBright){
		rMin = rMin < 1 ? 1 : rMin;
		this.cx = cx;
		this.cy = cy;
		this.rMin = rMin;
		this.rMax = rMax;
		this.numThetaSamples = numThetaSamples;
		this.cellsAreBright = cellsAreBright;
		this.rIncrement = rIncrement;
		this.originalImage = originalImage;
		
		this.paddingSize = (int) Math.round(numThetaSamples * (Constants.POLAR_PADDING_PERCENT/100));
		
		numRSamples = (int)Math.floor((rMax-rMin)/rIncrement)+1;
		polarImage = new PolarPixel[numRSamples][numThetaSamples+paddingSize*2];
		
		//First, calculate which pixels are touched by each of the concentric rings
		rings = new Ring[numRSamples];
		int rIndex = 0;
		for(double r = rMin; r <= rMax; r+=rIncrement){
			rings[rIndex] = new Ring(originalImage, cx, cy, r, cellsAreBright);
			rIndex++;
		}
		
		//Divide up each ring into segments of the specified theta increment (radians)
		double thetaIncrement = 2*Math.PI / numThetaSamples;
		double[] thetaValues = new double[numThetaSamples+1];
		for(int t=0; t < numThetaSamples+1; t++){
			thetaValues[t] = thetaIncrement * t;
		}
		
		//build the polar image
		rIndex = 0;
		for(double r = rMin; r <= rMax; r+=rIncrement){
			RingPixel[] ringPixels = rings[rIndex].ringPixels;
			int rp = 0;
			for(int t=paddingSize; t < numThetaSamples+paddingSize; t++){
				PolarPixel px = new PolarPixel();
				px.thetaStart = thetaValues[t-paddingSize];
				px.thetaEnd = thetaValues[t+1-paddingSize];
				px.r = r;

				//Polar pixel intensity will be a weighted average of the ring pixel
				//intensities. The weighting is based on how much of the ring is present
				//in the given pixel. 
				px.intensity = 0;
			
				boolean done = false;
				while(!done){
					if(px.thetaStart > ringPixels[rp].thetaMin && px.thetaEnd < ringPixels[rp].thetaMax){
						//polarPixel begins and ends in this ringPixel
						px.intensity += (px.thetaEnd - px.thetaStart) * ringPixels[rp].intensity;
						done = true;
					}
					else if(px.thetaStart > ringPixels[rp].thetaMin){
						//polarPixel begins in this ringPixel and continues on
						px.intensity += (ringPixels[rp].thetaMax - px.thetaStart) * ringPixels[rp].intensity;
						rp++;
					}
					else if(px.thetaEnd < ringPixels[rp].thetaMax){
						//polarPixel ends in this ringPixel
						px.intensity += (px.thetaEnd - ringPixels[rp].thetaMin) * ringPixels[rp].intensity;
						done = true;
					}
					else{
						//polarPixel goes through this ringPixel but does not begin or end in it
						//so use the whole thing
						px.intensity += (ringPixels[rp].thetaMax - ringPixels[rp].thetaMin) * ringPixels[rp].intensity;
						rp++;
					}
				}
				
				//OK, this polar pixel is ready! Add it to the polar image.
				polarImage[rIndex][t] = px;
				
				//also add the pixel into any padding locations it belongs in.
				if(t-paddingSize < paddingSize){
					//copy the leftmost part of the polar image to the padding on the right
					polarImage[rIndex][t+numThetaSamples] = px;
				}
				if(t+paddingSize >= (numThetaSamples+paddingSize)){
					//copy the rightmost part of the polar image to the padding on the left
					polarImage[rIndex][t-numThetaSamples] = px;
				}
			}
			
			rIndex++;
		}
		
		if(cellsAreBright){
			edgeLightToDarkFilter();
		}
		else{
			edgeDarkToLightFilter(); //optional -- runs an edge detection filter on the polar image
		}
        
		findEdge(); //populates the polarEdge variable with the traced path; dynamic programming step
        
		//The polar edge can be very jaggy and even discontinuous in Cartesian space. Fix that to populate cellEdge.
		makeConnectedEdge();
	}
	
	public double getMaxRadius(){
		double maxRadius = 0;
		for(int i = 0; i < polarEdge.size(); i++){
			if(polarEdge.get(i).r > maxRadius){
				maxRadius = polarEdge.get(i).r;
			}
		}
		return maxRadius;
	}

	
	public void makeConnectedEdge(){
		cellEdge = new ArrayList<CartesianPixel>();
		for(int i = 0; i < polarEdge.size(); i++){
			int j = (i+1) % polarEdge.size();
			cellEdge.addAll(connectPolarPixels(polarEdge.get(i), polarEdge.get(j)));
		}
	}
	
	public ArrayList<CartesianPixel> connectPolarPixels(PolarPixel a, PolarPixel b){
		//make a 4-connected path of CartesianPixels from the beginning of a to the beginning of b.
		ArrayList<CartesianPixel> path = new ArrayList<CartesianPixel>();
		
		CartesianPixel aStart = new CartesianPixel(a.r, a.thetaStart, cx, cy, originalImage.getWidth()-1, originalImage.getHeight()-1);
		CartesianPixel aEnd = new CartesianPixel(a.r, a.thetaEnd-0.001, cx, cy, originalImage.getWidth()-1, originalImage.getHeight()-1);
		CartesianPixel bStart = new CartesianPixel(b.r, b.thetaStart, cx, cy, originalImage.getWidth()-1, originalImage.getHeight()-1);

		path.add(aStart);
		connectPixels(aStart, aEnd, path);
		path.add(aEnd);
		connectPixels(aEnd, bStart, path);
		path.add(bStart);
		
		return path;
	}
	
	public void connectPixels(CartesianPixel a, CartesianPixel c, ArrayList<CartesianPixel> path){
		if(Math.abs(a.x - c.x) + Math.abs(a.y - c.y) <= 1){
			//these are already 4-connected, nothing else needs doing
			return; 
		}
		
		CartesianPixel b = new CartesianPixel(a,c,cx,cy, originalImage.getWidth()-1, originalImage.getHeight()-1);
		path.add(b);
		
		//recurse
		connectPixels(a,b,path);
		connectPixels(b,c,path);
	}


	private void findEdge(){
		//use dynamic programming to find the edge
		int numR = polarImage.length;
		int numTheta = polarImage[0].length;

		int[][] directionMatrix = new int[numR][numTheta];
		double[][] valueMatrix = new double[numR][numTheta];
		
		// determine the direction the edge path takes at each point
		for(int ti=numTheta-1; ti>=0; ti=ti-1){
			for(int ri=0; ri<numR; ri++){
				if(ti==numTheta-1){
					// rightmost column is just the pixel values
					valueMatrix[ri][ti] = polarImage[ri][ti].intensity;
					directionMatrix[ri][ti] = 0;
				}
				else{
					//Get the total for the three paths going to the right from this pixel
					double thisPixel = polarImage[ri][ti].intensity;
		            double rightTotal=valueMatrix[ri][ti+1] + thisPixel;
		            
		            double upRightTotal=0;
		            double downRightTotal=0;
		            if(ri==0){
		            	//only right and down-right paths exist
		            	downRightTotal=valueMatrix[ri+1][ti+1] + thisPixel;
		            }
		            else if(ri==numR-1){
		            	//only right and up-right paths exist
		            	upRightTotal=valueMatrix[ri-1][ti+1] + thisPixel;
		            }
		            else{
		            	//all three paths exist
		            	downRightTotal=valueMatrix[ri+1][ti+1] + thisPixel;
		            	upRightTotal=valueMatrix[ri-1][ti+1] + thisPixel;
		            }
		            
		            //compare the three paths; take the best.
		            //Default to straight if there is a tie.
		            if(rightTotal >= downRightTotal && rightTotal >= upRightTotal){
		            	directionMatrix[ri][ti] = 0;
		            	valueMatrix[ri][ti] = rightTotal;
		            }
		            else if(downRightTotal >= upRightTotal){
		            	//default to down if there's a tie between up and down.
		            	directionMatrix[ri][ti] = 1;
		            	valueMatrix[ri][ti] = downRightTotal;
		            }
		            else{
		            	//up-right is the winner!
		            	directionMatrix[ri][ti] = -1;
		            	valueMatrix[ri][ti] = upRightTotal;
		            }
				}				
			}
		}
		
		//Now find the edge path using valueMatrix and directionMatrix
		polarEdge = new ArrayList<PolarPixel>();
		
		//Start the path at the highest value of the leftmost column of valueMatrix
		int ri = 0;
		double maxValue = 0;
		for(int rStart=0; rStart<numR; rStart++){
			if(valueMatrix[rStart][0] > maxValue){
				ri = rStart;
				maxValue = valueMatrix[rStart][0];
			}
		}		

		if(paddingSize == 0){
			polarEdge.add(polarImage[ri][0]);
		}
		
		//Trace the path along directionMatrix.
		for(int ti=1; ti<numTheta; ti++){
			ri += directionMatrix[ri][ti-1];
			
			if(ti >= paddingSize && ti < numThetaSamples+paddingSize){
				polarEdge.add(polarImage[ri][ti]);
			}
		}
		
	}

	public int[][] getEdgePointsForRoi(){
		//The ImageJ ROI needs to get its points in a specific order to draw correctly.
		//Easiest way is to trace the edge using an edge-following algorithm.
		CartesianPixelPath path = new CartesianPixelPath(cellEdge);
		path.traceEdge();
		pixelCellEdge = path.getConnectedPath();
		
		ArrayList<Integer> edgePointsX = new ArrayList<Integer>();
		ArrayList<Integer> edgePointsY = new ArrayList<Integer>();
		
		int minX = 0;
		int minY = 0;
		
		for(int i=0; i<pixelCellEdge.size(); i++){
			Pixel px = pixelCellEdge.get(i);
			edgePointsX.add(px.x+1);
			edgePointsY.add(px.y+1);	
		}
		
		//convert to array
		int[][] edgePoints = new int[2][edgePointsX.size()];
		for(int i=0; i < edgePointsX.size(); i++){
			edgePoints[0][i] = edgePointsX.get(i) - minX;
			edgePoints[1][i] = edgePointsY.get(i) - minY;
		}
		return edgePoints;
	}

	public ArrayList<Pixel> getEdgePoints(){
		ArrayList<Pixel> edgePoints = new ArrayList<Pixel>();
		
		for(int i=0; i<polarEdge.size(); i++){
			PolarPixel px = polarEdge.get(i);
			double thetaIncrement = 1/(100*px.r*2*Math.PI);
			
			for(double theta = px.thetaStart; theta <= px.thetaEnd; theta += thetaIncrement){
				//x
				int x = cx;
				double cosTheta = Math.cos(theta);
				if(cosTheta < 0){
					x += (int) Math.ceil(px.r*cosTheta);
				}
				else{
					x += (int) Math.floor(px.r*cosTheta);
				}
					
				//y
				int y = cy;
				double sinTheta = Math.sin(theta);
				if(sinTheta < 0){
					y += (int) Math.ceil(px.r*sinTheta);
				}
				else{
					y += (int) Math.floor(px.r*sinTheta);
				}
				Pixel edgePoint = new Pixel(x,y);
				edgePoints.add(edgePoint);
			}
		}
		
		//remove duplicate points
		Collections.sort(edgePoints, new Comparator<Pixel>() {
		    public int compare(Pixel p, Pixel q) {
		    	if(p.y < q.y){
		    		return -1;
		    	}
		    	else if(p.y > q.y){
		    		return 1;
		    	}
		    	else{
		    		if(p.x < q.x){
			    		return -1;
			    	}
		    		else if(p.x > q.x){
		    			return 1;
		    		}
		    		else{
		    			return 0;
		    		}
		    	}
		    }
		});
		for(int i = 1; i < edgePoints.size(); i++){
			if(edgePoints.get(i).x == edgePoints.get(i-1).x && edgePoints.get(i).y == edgePoints.get(i-1).y ){
				edgePoints.remove(i);
				i--;
			}
		}
		
		return edgePoints;
	}

	public void edgeDarkToLightFilter(){
		for(int ri = numRSamples-1; ri >= 0; ri--){
			for(int t=0; t < numThetaSamples; t++){
				if(ri==0){
					polarImage[ri][t].intensity = 0;
				}
				else{
					polarImage[ri][t].intensity -= polarImage[ri-1][t].intensity;
					polarImage[ri][t].r -= rIncrement/2;
				}
			}
		}
	}

	public void edgeLightToDarkFilter(){
		for(int ri = 0; ri < numRSamples; ri++){
			for(int t=0; t < numThetaSamples; t++){
				if(ri==numRSamples-1){
					polarImage[ri][t].intensity = 0;
				}
				else{
					polarImage[ri][t].intensity -= polarImage[ri+1][t].intensity;
					polarImage[ri][t].r += rIncrement/2;
				}
			}
		}
	}

	public ArrayList<Pixel> getPixelCellEdge(){
		return pixelCellEdge;
	}

	public ArrayList<CartesianPixel> getCartesianCellEdge(){
		return cellEdge;
	}
	
	public static void print(String str){
		System.out.println(str);
	}
	
	
	/* The functions below are used only for debugging and displaying how the algorithm works */
	
	
	public int[][] getEdgePointsMagnified(int magFactor){
		//This is a function that's used to make visualizations of how the polar transform 
		//algorithm works. A magFactor of around 10 is pretty good. 
		ArrayList<Integer> edgePointsX = new ArrayList<Integer>();
		ArrayList<Integer> edgePointsY = new ArrayList<Integer>();
		
		for(int i=0; i<polarEdge.size(); i++){
			PolarPixel px = polarEdge.get(i);
			double thetaIncrement = 1/(100*px.r*2*Math.PI);
			
			for(double theta = px.thetaStart; theta <= px.thetaEnd; theta += thetaIncrement){
				int x = (int) Math.round(px.r*magFactor*Math.cos(theta));
				int y = (int) Math.round(px.r*magFactor*Math.sin(theta));
				edgePointsX.add(x + cx*magFactor);
				edgePointsY.add(y + cy*magFactor);
			}
		}
		
		//convert to array
		int[][] edgePoints = new int[2][edgePointsX.size()];
		for(int i=0; i < edgePointsX.size(); i++){
			edgePoints[0][i] = edgePointsX.get(i)+5; //the +5 allows it to align to the pixel grid nicely
			edgePoints[1][i] = edgePointsY.get(i)+5;
		}
		return edgePoints;
	}


	public ImagePlus getPolarImage(){
		int numR = polarImage.length;
		int numTheta = polarImage[0].length;
		
		ShortProcessor sp = new ShortProcessor(numTheta, numR);
		for(int ri=0; ri < numR; ri++){
			for(int ti=0; ti < numTheta; ti++){
				sp.set(ti, ri, (short)Math.round(polarImage[ri][ti].intensity));
			}
		}
		
		ImagePlus ip = new ImagePlus("Polar Transform",sp);
		return ip;
	}

	
}
