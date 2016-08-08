import cellMagicWand.CommandLine;

public class MagicWand {

    public static void main(String[] args) {
	String photo_dir = args[0];
	int num_photos = Integer.parseInt(args[1]);
	String[] photo_names = val_names(photo_dir, num_photos, "stk", ".tif");
	String params_dir = args[2];
	String[] params_names = val_names(params_dir, num_photos, "params", ".txt");
	String[] edges_names = val_names(params_dir, num_photos, "edges", ".txt");
	
	assert(photo_names.length == params_names.length && photo_names.length == edges_names.length);

	for(int i = 0; i < photo_names.length; i++) {
	    String[] a = new String[3];
	    a[0] = photo_names[i];
	    a[1] = params_names[i];
	    a[2] = edges_names[i];
	    try {
		CommandLine cl = new CommandLine(a);
	    } catch (Exception e) {}
	}
    }

    private static String[] val_names(String dir, int num, String prefix, String ext) {
	//String[] nums = {"1","3","19","24","26"};
	//String[] nums = {"2","7","16","25","28"};
	//String[] nums = {"1","2","3","7","16","19","24","25","26","28"};
	//String[] nums = {"4", "6", "7", "9", "10"};
	String[] nums = {"3", "19", "26"};

	String[] names = new String[3];
	for(int i = 0; i < 3; i++) {
	    names[i] = dir + prefix + nums[i] + ext;
	}
	return names;
    
    }

    private static String[] make_names(String dir, int num, String prefix, String ext) {
	String[] names = new String[num];
	for(int i = 0; i < num; i++) {
	    names[i] = dir + prefix + (i+1) + ext;
	}
	return names;
    }
}