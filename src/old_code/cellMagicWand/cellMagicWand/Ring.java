// Theo Walker
// Max Planck Florida Institute
// naroom@gmail.com

package cellMagicWand;

import java.util.ArrayList;

import ij.ImagePlus;


public class Ring {
	//Visualize drawing a circle through the pixels of the original image. 
	//each RingPixel corresponds to a pixel that the circle runs through.
	//We record the range of theta values that tell us how long the arc through
	//that pixel is. This theta range will be used to scale intensity when we 
	//construct the polar image.

	RingPixel[] ringPixels; 
	double r;
	boolean cellsAreBright;
	
	public Ring(ImagePlus originalImage, int cx, int cy, double r, boolean cellsAreBright){
		//Creates a 4-connected set of pixels that compose a circle of fixed radius. 
		//Run RingTest to see what each ring looks like on an image.
		this.r = r;
		this.cellsAreBright = cellsAreBright;
		double r2 = Math.pow(r,2); //precalculate r^2, it will be used many times
		
		//Start the circle at (0,r) initially.
		int y = 0;
		int x = (int)Math.round(r-0.0001);

		// At what value of theta did the arc through this pixel start and end?
		double thetaIn = Math.atan2(y-0.5, Math.sqrt(r2 - Math.pow(y-0.5,2))) % (2*Math.PI);
		double thetaOut = 0; //will fill this in later

		//Strategy: We will calculate the first-quadrant arc of the circle, then replicate it to make the other quadrants.
		//Hooray for symmetry!
		ArrayList<RingPixel> firstQuadrant = new ArrayList<RingPixel>(); 
		
		while(x >= 0){
			//For this value of x, find the lowest and highest value of y that the circle falls on
	        double yLow = (Math.sqrt(r2 - Math.pow(x+0.5,2)));
	        double yHigh = (Math.sqrt(r2 - Math.pow(x-0.5,2)));
	        
	        //start at the low value, in case we're not there already
	        while ((y+0.5) < yLow){
	        	y=y+1; 
	        }
	        
	        //Continue until we hit the high value
	        while ((y-0.5)<=yHigh){
				if((y+0.5) < yHigh){
				    //The line exits out the top of this pixel
				    double yExitPoint = y+0.5;
					double xExitPoint = Math.sqrt(r2 - Math.pow(yExitPoint,2));
				    thetaOut = Math.atan2(yExitPoint, xExitPoint);
				}
				else{
				    //The line exits out the left side of this pixel
					double xExitPoint = x-0.5;
					double yExitPoint = Math.sqrt(r2 - Math.pow(xExitPoint,2));
				    thetaOut = Math.atan2(yExitPoint, xExitPoint);
				}
				
				//Make a RingPixel and add it
				RingPixel p = new RingPixel();
				p.x = cx+x;
				p.y = cy+y;
				p.thetaMin = thetaIn;
				p.thetaMax = thetaOut;
				p.r = r;
				p.intensity = getPixel(originalImage,p.x, p.y); 
				firstQuadrant.add(p);
				
				// advance to next pixel
				thetaIn = thetaOut;
				y=y+1;
	        }

	        y=y-1;
	        x=x-1;
	    }
		    
		//OK, now we have the first quadrant. Make the other 3.
		//The top, left, and bottom pixels will be repeated, so we will avoid adding them twice.
		//We will, however, add a copy of the first pixel at the end -- saves us some hassle later.
		ringPixels = new RingPixel[firstQuadrant.size()*4-3];
		
		//Fill in Q1 (already calculated)
		int rpIndex = 0;
		for(int i = 0; i < firstQuadrant.size(); i++){
			ringPixels[rpIndex] = firstQuadrant.get(i);
			rpIndex++;
		}

		//Q2 is the reflection of Q1 across the Y-axis, skipping the last point of Q1.
		for(int i = firstQuadrant.size()-2; i>=0 ; i--){
			RingPixel q = firstQuadrant.get(i);
			RingPixel p = new RingPixel();
			p.x = cx-(q.x-cx); //flip to other side (hint: x = q.x-cx, so cx-(q.x-cx) equals our desired cx-x.)
			p.y = q.y;
			p.thetaMin = Math.PI - q.thetaMax;
			p.thetaMax = Math.PI - q.thetaMin;
			p.r = q.r;
			p.intensity = getPixel(originalImage,p.x, p.y);
			ringPixels[rpIndex] = p;
			rpIndex++;
		}
		//Q3 is the reflection of Q1 across both axes, skipping the last point of Q2.
		for(int i = 1; i < firstQuadrant.size(); i++){
			RingPixel q = firstQuadrant.get(i);
			RingPixel p = new RingPixel();
			p.x = cx-(q.x-cx); //this time 
			p.y = cy-(q.y-cy); //we flip both
			p.thetaMin = q.thetaMin + Math.PI;
			p.thetaMax = q.thetaMax + Math.PI;
			p.r = q.r;
			p.intensity = getPixel(originalImage,p.x, p.y); 
			ringPixels[rpIndex] = p;
			rpIndex++;
		}

		//Q4 is the reflection of Q1 across the X-axis, 
		//skipping the last point of Q3 as well as the first point of Q1.
		for(int i = firstQuadrant.size()-2; i>=1; i--){
			RingPixel q = firstQuadrant.get(i);
			RingPixel p = new RingPixel();
			p.x = q.x; 
			p.y = cy-(q.y-cy); //flip Y
			p.thetaMin = 2*Math.PI - q.thetaMax;
			p.thetaMax = 2*Math.PI - q.thetaMin;
			p.r = q.r;
			p.intensity = getPixel(originalImage,p.x, p.y); 
			ringPixels[rpIndex] = p;
			rpIndex++;
		}
		
		//add the first pixel to the end as well
		RingPixel q = firstQuadrant.get(0);
		RingPixel p = new RingPixel();
		p.x = q.x; 
		p.y = q.y;
		p.thetaMin = 2*Math.PI + q.thetaMin;
		p.thetaMax = 2*Math.PI + q.thetaMax;
		p.r = q.r;
		p.intensity = getPixel(originalImage,p.x, p.y); 
		ringPixels[rpIndex] = p;
	}

	public double getPixel(ImagePlus img, int x, int y){
		//this overrides some behaviors of ImagePlus.getPixel() that we don't like. 
		//Specifically, ImagePlus.getPixel() returns 0 for anything that's off the image
		//e.g. ImagePlus.getPixel(-1,-1) returns 0.
		//This does bad things to our edge detection algorithms, because it can detect a strong
		//gradient along the edge where there isn't really one. Instead, we just return the value of the 
		//closest pixel, which fixes things in most cases.
		
		if(x < 0){
			x = 0;
		}
		if(x > img.getWidth()-1){
			x = img.getWidth()-1;
		}
		if(y < 0){
			y = 0;
		}
		if(y > img.getHeight()-1){
			y = img.getHeight()-1;
		}
		
		return img.getPixel(x, y)[0];
	}
	
}
