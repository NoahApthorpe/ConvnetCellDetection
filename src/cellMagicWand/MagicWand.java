import cellMagicWand.CommandLine;
import java.io.File;
import java.util.ArrayList;

public class MagicWand {

    public static void main(String[] args) {
	String photo_dir = args[0];
	String params_dir = args[1];
	File photo_dir_f = new File(photo_dir);
	File params_dir_f = new File(params_dir);

	File[] params_names_f = params_dir_f.listFiles();
	ArrayList<String> photo_names = new ArrayList<String>();
	ArrayList<String> params_names = new ArrayList<String>();
	ArrayList<String> edges_names = new ArrayList<String>();
	for (File f : params_names_f) {
		try{
			String name = f.getName();
			String name_noext = name.substring(0, name.lastIndexOf('.'));
			String ext = name.substring(name.lastIndexOf('.'), name.length());
			if (!ext.equals(".txt")) continue;
			photo_names.add(photo_dir + name_noext + ".tif");
			params_names.add(params_dir + name_noext + ".txt");
			edges_names.add(params_dir + name_noext + "_edges.txt");
		} catch (IndexOutOfBoundsException e) {e.printStackTrace();}
	}
	
	assert(photo_names.size() == params_names.size() && photo_names.size() == edges_names.size());

	for(int i = 0; i < photo_names.size(); i++) {
	    String[] a = new String[3];
	    a[0] = photo_names.get(i);
	    a[1] = params_names.get(i);
	    a[2] = edges_names.get(i);
	    //try {
		CommandLine cl = new CommandLine(a);
	    //} catch (Exception e) {e.printStackTrace();}
	}
    }
}