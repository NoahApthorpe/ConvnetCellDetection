// Theo Walker
// Max Planck Florida Institute
// naroom@gmail.com

package cellMagicWand;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;

import ij.ImagePlus;
import ij.io.Opener;
//handles the running of Cell Magic Wand via command line
import cellMagicWand.Constants;

public class CommandLine {

	int minDiameter = Constants.DEFAULT_MIN_DIAMETER;
	int maxDiameter = Constants.DEFAULT_MAX_DIAMETER;
	double circumferenceSampleRate = Constants.DEFAULT_CIRCUMFERENCE_SAMPLE_RATE;
	boolean cellsAreBright = true;
	
	ArrayList<Integer> seedsX = new ArrayList<Integer>();
	ArrayList<Integer> seedsY = new ArrayList<Integer>();
	ImagePlus imp;
	
	public CommandLine(String[] args){
		//parse args, populate member variables
		if(args.length != 3){
			printHelp();
			return;
		}

		//open image file
		print("");
		print("imagePath: " + args[0]);
		try{
	        Opener o = new Opener();
	        imp = o.openImage(args[0]);
	        if(imp == null){
	        	throw new Exception("Wrong file!");
	        }
		}
		catch(Exception ex){
			print("Error reading file " + args[0]);
			return;
		}
		
        //parse params file
		String paramsPath = args[1];
		print("paramsPath: " + paramsPath);
		String line = "";
		try{
			BufferedReader br = new BufferedReader(new FileReader(paramsPath));
			while ((line = br.readLine()) != null) {
				String[] toks = line.split(" ");
				if(toks[0].equalsIgnoreCase("minDiameter")){
					minDiameter = (int) Double.parseDouble(toks[1]);
				}
				else if(toks[0].equalsIgnoreCase("maxDiameter")){
					maxDiameter = (int) Double.parseDouble(toks[1]);
				}
				else if(toks[0].equalsIgnoreCase("roughness")){
					circumferenceSampleRate = Double.parseDouble(toks[1]);
				}
				else if(toks[0].equalsIgnoreCase("imageType")){
					cellsAreBright = toks[1].equalsIgnoreCase("brightCells");
				}
				else if(toks[0].equalsIgnoreCase("seed")){
					int seedX = Integer.parseInt(toks[1]);
					int seedY = Integer.parseInt(toks[2]);
					seedsX.add(seedX);
					seedsY.add(seedY);					
				}
			}
			print("Min diameter: " + minDiameter);
			print("Max diameter: " + maxDiameter);
			print("roughness: " + circumferenceSampleRate);
			print("Seed points: ");
			for(int i = 0; i < seedsX.size(); i++){
				print("(" + seedsX.get(i) + "," + seedsY.get(i) + ")");
			}
			br.close();
		}
		catch(Exception ex){
			ex.printStackTrace();
			print("");
			print("Error reading file " + paramsPath);
			if(!line.isEmpty()){
				print("at line: " + line);
			}
			return;
		}
		print("");
		
		//run polar transform for each seed
		ArrayList< ArrayList<Pixel> > pixelCellEdges = new ArrayList< ArrayList<Pixel> >();
		for(int i = 0; i < seedsX.size(); i++){
		    try{
		        print("Running point " + (i+1) + " of " + seedsX.size());
			int rMin = (int) Math.round(minDiameter/2);
			int rMax = (int) Math.round(maxDiameter/2);
			
			//Run the polar transform once to find the cell's actual radius, so we can translate circumferenceSampleRate to numThetaSamples
			PolarTransform radiusFind = new PolarTransform(imp, seedsX.get(i), seedsY.get(i), rMin, rMax, Constants.PIXELS_PER_R_SAMPLE, Constants.THETA_SAMPLES_FOR_RADIUS_FINDING, cellsAreBright);
			double maxRadius = radiusFind.getMaxRadius();

			print("meanRadius: " + maxRadius);
			int numThetaSamples = (int)Math.round(2*Math.PI*maxRadius*circumferenceSampleRate);
			if(numThetaSamples < 2)
				numThetaSamples = 2;
			print("Theta samples: " + numThetaSamples);
			
			//Now do the transform used in ROI finding
			PolarTransform pt = new PolarTransform(imp, seedsX.get(i), seedsY.get(i), rMin, rMax, Constants.PIXELS_PER_R_SAMPLE, numThetaSamples, cellsAreBright);
			
			ArrayList<Pixel> cellEdge = pt.getEdgePoints();
			pixelCellEdges.add(cellEdge);
		    } catch (Exception e) {e.printStackTrace();}
		}
		
		//make outfile
		try{
			BufferedWriter bw = new BufferedWriter(new FileWriter(args[2]));
			for(int i = 0; i < pixelCellEdges.size(); i++){
				for(int j = 0; j < pixelCellEdges.get(i).size(); j++){
					bw.write("(" + pixelCellEdges.get(i).get(j).x + ","+ pixelCellEdges.get(i).get(j).y + ")");
					if(j != pixelCellEdges.get(i).size()-1){
						bw.write(",");
					}
					else{
						bw.write("\n");
					}
				}
			}
			bw.close();
		}
		catch(Exception ex){
			ex.printStackTrace();
			print("Error writing to file " + args[2]);
			return;
		}
	}
	
    public void printHelp(){
        print("Command line arguments: ");	
        print("");
        print(Constants.PROGRAM_NAME + " imageFilePath paramFilePath outFilePath");
        print("");
        print("Example:");
        print(Constants.PROGRAM_NAME + " C:/image.tif C:/params.txt C:/cellEdges.txt");
        print("");
        print("Parameter file sample:");
        print("==============");
        print("minDiameter 8");
        print("maxDiameter 250");
        print("roughness 5.0");
        print("imageType brightCells");
        print("seed 121 259");
        print("seed 244 250");
        print("seed 60 603");
        print("seed 563 235");
        print("==============");
        print("Note that seed points are (X,Y) coordinates. The imageType can be 'brightCells' or 'darkCells'.");
        return;
    }
    
	public void print(String s){
		System.out.println(s);
	}
}
//Theo Walker
//Max Planck Florida Institute
//naroom@gmail.com

