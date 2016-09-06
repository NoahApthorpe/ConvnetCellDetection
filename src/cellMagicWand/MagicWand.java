import cellMagicWand.CommandLine;
import java.io.File;
import java.util.ArrayList;

public class MagicWand {

    public static void main(String[] args) {
	String photo_dir = args[0];
	String params_dir = args[1];
	File photo_dir_f = new File(photo_dir);
	File params_dir_f = new File(params_dir);

	boolean is_labeled = photo_dir_f.getName().equals("labeled");
	File training_dir = new File(photo_dir + "training/");
	File validation_dir = new File(photo_dir + "validation/");
	File test_dir = new File(photo_dir + "test/");
	File[] training_files = new File[0];
	File[] validation_files = new File[0];
	File[] test_files = new File[0];
	if (is_labeled && training_dir.isDirectory()) training_files = training_dir.listFiles();
	if (is_labeled && validation_dir.isDirectory()) validation_files = validation_dir.listFiles();
	if (is_labeled && test_dir.isDirectory()) test_files = test_dir.listFiles();
	
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
			if (is_labeled) {
			    if (inFileArray(training_files, name_noext)) {
				
				photo_names.add(photo_dir + "training/" + name_noext + ".tif");
			    } else if (inFileArray(validation_files, name_noext)) {
				photo_names.add(photo_dir + "validation/" + name_noext + ".tif");
			    } else if (inFileArray(test_files, name_noext)) {
				photo_names.add(photo_dir + "test/" + name_noext + ".tif");
			    }
			} else {
			    photo_names.add(photo_dir + name_noext + ".tif");
			}
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

    private static boolean inFileArray(File[] fileArray, String name) {
	for (File f : fileArray) {
	    String fname = f.getName();
	    String fname_noext = fname.substring(0, fname.lastIndexOf('.'));
		
		if (fname_noext.equals(name)) return true;
	}
	return false;
    }
}
