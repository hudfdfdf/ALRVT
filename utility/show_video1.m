function update_visualization_func = show_video(img_files, video_path, resize_image)
%SHOW_VIDEO
%   Visualizes a tracker in an interactive figure, given a cell array of
%   image file names, their path, and whether to resize the images to
%   half size or not.
%
%   This function returns an UPDATE_VISUALIZATION function handle, that
%   can be called with a frame number and a bounding box [x, y, width,
%   height], as soon as the results for a new frame have been calculated.
%   This way, your results are shown in real-time, but they are also
%   remembered so you can navigate and inspect the video afterwards.
%   Press 'Esc' to send a stop signal (returned by UPDATE_VISUALIZATION).
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/


	%store one instance per frame
	num_frames = numel(img_files);
	boxes = cell(num_frames,1);
    boxes_sw = cell(num_frames,1);
   boxes2=[];
   boxes3=[];
	%create window
	[fig_h, axes_h, unused, scroll] = videofig(num_frames, @redraw, [], [], @on_key_press);  %#ok, unused outputs
	set(fig_h, 'Name', ['Tracker - ' video_path])
	axis off;
	
	%image and rectangle handles start empty, they are initialized later
	im_h = [];
	rect_h = [];
    rect_h1 = [];
	  rect_h2 = [];
        rect_h3 = [];
	update_visualization_func = @update_visualization;
	stop_tracker = false;
	

	function stop = update_visualization(frame, box,sw,box2,box3)
		%store the tracker instance for one frame, and show it. returns
		%true if processing should stop (user pressed 'Esc').
		boxes{frame} = box;
        boxes_sw{frame} = sw;
        boxes2=box2;
        boxes3=box3;
		scroll(frame);
		stop = stop_tracker;
	end

	function redraw(frame)
		%render main image
		im = imread([video_path img_files{frame}]);
%         im = imread([video_path img_files{1}]);
%         im=circshift(im,[5*frame,5*frame]);
        
		if isempty(im_h),  %create image
			im_h = imshow(im, 'Border','tight', 'InitialMag',200, 'Parent',axes_h);
		else  %just update it
			set(im_h, 'CData', im)
		end
		
		%render target bounding box for this frame
		if isempty(rect_h),  %create it for the first time
			rect_h = rectangle('Position',[0,0,1,1], 'EdgeColor','g', 'Parent',axes_h);
        end
        if isempty(rect_h1),  %create it for the first time
			rect_h1 = rectangle('Position',[0,0,1,1], 'EdgeColor','b', 'Parent',axes_h);
        end
         if isempty(rect_h2),  %create it for the first time
			rect_h2 = rectangle('Position',[0,0,1,1], 'EdgeColor','r', 'Parent',axes_h);
		end
         if isempty(rect_h3),  %create it for the first time
			rect_h3 = rectangle('Position',[0,0,1,1], 'EdgeColor','y', 'Parent',axes_h);
		end
        
		if ~isempty(boxes{frame}),
			set(rect_h, 'Visible', 'on', 'Position', boxes{frame});
            set(rect_h1, 'Visible', 'on', 'Position', boxes_sw{frame});
          set(rect_h2, 'Visible', 'on', 'Position', boxes2);
            set(rect_h3, 'Visible', 'on', 'Position', boxes3);
		else
			set(rect_h, 'Visible', 'off');
		end
	end

	function on_key_press(key)
		if strcmp(key, 'escape'),  %stop on 'Esc'
			stop_tracker = true;
		end
	end

end

