function [img_files, pos, target_sz, ground_truth, video_path] = load_video_info_vot(base_path, video)
%LOAD_VIDEO_INFO
%   Loads all the relevant information for the video in the given path:
%   the list of image files (cell array of strings), initial position
%   (1x2), target size (1x2), the ground truth information for precision
%   calculations (Nx2, for N frames), and the path where the images are
%   located. The ordering of coordinates and sizes is always [y, x].
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/


	%see if there's a suffix, specifying one of multiple targets, for
	%example the dot and number in 'Jogging.1' or 'Jogging.2'.
sequence_name = video;    

% path to the folder with VOT sequences
base_path = fullfile(base_path, sequence_name);
% img_dir = dir(fullfile(base_path, '*.jpg'));
video_path=[base_path,'/'];
% initialize bounding box - [x,y,width, height]
ground_truth = read_vot_regions(fullfile(base_path, 'groundtruth.txt'));
	
	%set initial position and size
	target_sz = [ground_truth(1,4), ground_truth(1,3)];
	pos = [ground_truth(1,2), ground_truth(1,1)] + floor(target_sz/2);
	
	if size(ground_truth,1) == 1,
		%we have ground truth for the first frame only (initial position)
		ground_truth = [];
	else
		%store positions instead of boxes
		ground_truth = ground_truth(:,[2,1]) + ground_truth(:,[4,3]) / 2;
	end
	
	
	%from now on, work in the subfolder where all the images are
% 	video_path = [video_path 'img/'];
	
	%for these sequences, we must limit ourselves to a range of frames.
	%for all others, we just load all png/jpg files in the folder.'David', 300, 770;'diving', 1, 215
	frames = {'Football1', 1, 74;
			  'Freeman3', 1, 460;
			  'Freeman4', 1, 283};

	idx = find(strcmpi(video, frames(:,1)));

	if isempty(idx),
		%general case, just list all images
		img_files = dir(fullfile(base_path, '*.png'));
		if isempty(img_files),
			img_files = dir(fullfile(base_path, '*.jpg'));
			assert(~isempty(img_files), 'No image files to load.')
		end
		img_files = sort({img_files.name});
	else
		%list specified frames. try png first, then jpg.
		if exist(sprintf('%s%04i.png', base_path, frames{idx,2}), 'file'),
			img_files = num2str((frames{idx,2} : frames{idx,3})', '%04i.png');
			
		elseif exist(sprintf('%s%04i.jpg', base_path, frames{idx,2}), 'file'),
			img_files = num2str((frames{idx,2} : frames{idx,3})', '%04i.jpg');
			
		else
			error('No image files to load.')
		end
		
		img_files = cellstr(img_files);
	end
	
end

