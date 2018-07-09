
% run_tracker.m

close all;
% clear all;

%choose the path to the videos (you'll be able to choose one with the GUI)
base_path = 'sequences/';

% HOG feature parameters
hog_params.nDim = 31;

% Grayscale feature parameters
grayscale_params.colorspace='gray';
grayscale_params.nDim = 1;
% Color name feature papameters
temp = load('w2crs');
colorname_params.w2c = temp.w2crs;
colorname_params.nDim =10;
% Global feature parameters 
params.t_features = {
    struct('getFeature',@get_colorspace, 'fparams',grayscale_params),...  % Grayscale is not used as default
    ...struct('getFeature',@get_colorname,'fparams',colorname_params),...
    struct('getFeature',@get_fhog,'fparams',hog_params),...
};
% Global feature parameters
params.t_global.cell_size = 4;                  % Feature cell size
params.t_global.cell_selection_thresh = 0.75^2; % Threshold for reducing the cell size in low-resolution cases
params.t_global.normalize_power = 2;    % Lp normalization with this p
params.t_global.normalize_size = true;  % Also normalize with respect to the spatial size of the feature
params.t_global.normalize_dim = true;   % Also normalize with respect to the dimensionality of the feature

% Filter parameters
params.search_area_shape = 'square';    % the shape of the training/detection window: 'proportional', 'square' or 'fix_padding'
params.search_area_scale = 4.0;         % the size of the training/detection area proportional to the target size
params.filter_max_area = 50^2;          % the size of the training/detection area in feature grid cells

% Learning parameters
params.padding = 1.0;         			% extra area surrounding the target
params.output_sigma_factor = 1/16;		% standard deviation for the desired translation filter output
params.scale_sigma_factor = 1/4;        % standard deviation for the desired scale filter output
params.lambda = 1e-2;					% regularization weight (denoted "lambda" in the paper)
params.learning_rate = 0.025;			% tracking model learning rate (denoted "eta" in the paper)
params.number_of_scales = 7;           % number of scale levels (denoted "S"=7 in the paper)
params.scale_step = 1.01;               % Scale increment factor (denoted "a" in the paper)
params.scale_model_max_area = 512;      % the maximum size of scale examples
params.init_strategy = 'indep';         % strategy for initializing the filter: 'const_reg' or 'indep'
params.num_GS_iter = 4;                 % number of Gauss-Seidel iterations in the learning

% Detection parameters
params.refinement_iterations = 1;       % number of iterations used to refine the resulting position in a frame
params.interpolate_response = 4;        % correlation score interpolation strategy: 0 - off, 1 - feature grid, 2 - pixel grid, 4 - Newton's method
params.newton_iterations = 5;           % number of Newton's iteration to maximize the detection scores

% Debug and visualization
params.visualization = 1;
params.debug = 0;

%ask the user for the video
video_path = choose_video(base_path);
if isempty(video_path), return, end  %user cancelled
[img_files, pos, target_sz, ground_truth, video_path] = ...
	load_video_info(video_path);

params.init_pos = floor(pos) + floor(target_sz/2);
params.wsize = floor(target_sz);
params.img_files = img_files;
params.video_path = video_path;

% [positions, fps] = dsst_L1(params);
[positions, fps] = MS_L1CFT(params);
% calculate precisions
[distance_precision, PASCAL_precision, average_center_location_error,average_overlap_rate] = ...
    compute_performance_measures(positions, ground_truth,video_path);

fprintf('Center Location Error: %.3g pixels\n Overlap Rate: %.3g \n Distance Precision: %.3g %%\nOverlap Precision: %.3g %%\nSpeed: %.3g fps\n', ...
    average_center_location_error, average_overlap_rate,100*distance_precision, 100*PASCAL_precision, fps);