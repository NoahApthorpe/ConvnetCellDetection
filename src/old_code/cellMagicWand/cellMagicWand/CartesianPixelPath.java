// Theo Walker
// Max Planck Florida Institute
// naroom@gmail.com

package cellMagicWand;

import java.util.ArrayList;

public class CartesianPixelPath {
	
	ArrayList<CartesianPixel> originalPath;
	ArrayList<Pixel> connectedPath;
	public enum Direction {UP, DOWN, LEFT, RIGHT, ERROR};
	
	
	public CartesianPixelPath(ArrayList<CartesianPixel> path_){
		this.originalPath = path_;		
		connectedPath = new ArrayList<Pixel>();	
	}
	
	public void traceEdge(){
		int xStart = 0;
		int yStart = 0;
		for(int i = 0; i < originalPath.size(); i++){
			if(originalPath.get(i).x > xStart){
				xStart = originalPath.get(i).x;
				yStart = originalPath.get(i).y;
			}
		}

		int x = xStart;
		int y = yStart;
		do{
			Direction d = moveAlongPath(x,y);
			if(d == Direction.LEFT){
				x=x-1;
			}
			else if(d == Direction.RIGHT){
				x=x+1;
			}
			else if(d == Direction.UP){
				y=y-1;
			}
			else if(d == Direction.DOWN){
				y=y+1;
			}
			else if(d == Direction.ERROR){
				x = xStart;
				y = yStart;
			}
			
			connectedPath.add(new Pixel(x,y));
			
		}while(!(x == xStart && y == yStart));
	}
		
	private Direction moveAlongPath(int x, int y){
		// This implements a basic edge following algorithm. 
		// It can tell where to go next to follow the path, 
		// given the information in a 2-by-2 box of pixels.
		
		//(x,y) specifies the top left corner of the box we are evaluating.
		int boxType = 0; //binary 0000
		if(isInEdge(x,y)){
			boxType = boxType + 8; //binary 1000
		}
		if(isInEdge(x+1,y)){
			boxType = boxType + 4; //binary 0100
		}
		if(isInEdge(x,y+1)){
			boxType = boxType + 2; //binary 0010
		}
		if(isInEdge(x+1,y+1)){
			boxType = boxType + 1; //binary 0001
		}
		
		//now, of the 16 possibilities, there's no way to reach a 0000 or 1111 state.
		if(boxType == 0 || boxType == 15){
			return Direction.ERROR;
		}

		//Also, it's impossible to have a diagonal (0110 or 1001).
		//We wouldn't know which direction to go from such a thing anyway, 
		//at least not without knowing the previous move.
		if(boxType == 6 || boxType == 9){
			return Direction.ERROR;
		}
		
		//For the diagrams, 'x' indicates an edge pixel and 'o' indicates a non-edge pixel.

		if(boxType == 1){
			// [oo]  
			// [ox] 
			return Direction.RIGHT;
		}
		else if(boxType == 2){
			// [oo]  
			// [xo] 
			return Direction.DOWN;
		}
		else if(boxType == 3){
			// [oo]  
			// [xx] 
			return Direction.RIGHT;
		}
		else if(boxType == 4){
			// [ox]  
			// [oo] 
			return Direction.UP;
		}
		else if(boxType == 5){
			// [ox]  
			// [ox] 
			return Direction.UP;
		}
		else if(boxType == 7){
			// [ox]  
			// [xx] 
			return Direction.UP;
		}
		else if(boxType == 8){
			// [xo]  
			// [oo] 
			return Direction.LEFT;
		}
		else if(boxType == 10){
			// [xo]  
			// [xo] 
			return Direction.DOWN;
		}
		else if(boxType == 11){
			// [xo]  
			// [xx] 
			return Direction.RIGHT;
		}
		else if(boxType == 12){
			// [xx]  
			// [oo] 
			return Direction.LEFT;
		}
		else if(boxType == 13){
			// [xx]  
			// [ox] 
			return Direction.LEFT;
		}
		else if(boxType == 14){
			// [xx]  
			// [xo] 
			return Direction.DOWN;
		}
		else{
			return Direction.ERROR;
		}
	}
	
	private boolean isInEdge(int x, int y){
		//returns true if pixel is part of original path
		//doing this the shitty way at first; can make it a binary search or hash later.
		for(int i = 0; i < originalPath.size(); i++){
			if(originalPath.get(i).x == x && originalPath.get(i).y == y){
				return true;
			}
		}
		return false;
	}
	
	public ArrayList<Pixel> getConnectedPath(){
		return connectedPath;
	}
	public static void print(String str){
		System.out.println(str);
	}
}
