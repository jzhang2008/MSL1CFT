function [positions, fps] = MS_L1CFT(params)

% [positions, fps] = dsst(params)

% parameters
search_area_scale = params.search_area_scale;
% padding = params.padding;                         	%extra area surrounding the target
output_sigma_factor = params.output_sigma_factor;	%spatial bandwidth (proportional to target)
% lambda = params.lambda;
learning_rate = params.learning_rate;
nScales = params.number_of_scales;
scale_step = params.scale_step;
% scale_sigma_factor = params.scale_sigma_factor;
% scale_model_max_area = params.scale_model_max_area;
refinement_iterations = params.refinement_iterations;
filter_max_area = params.filter_max_area;
interpolate_response = params.interpolate_response;
% num_GS_iter = params.num_GS_iter;
features = params.t_features;

% video_path = params.video_path;
img_files = params.img_files;
pos = floor(params.init_pos);
target_sz = floor(params.wsize);

debug = params.debug;
visualization = params.visualization || debug;
%visualization = params.visualization;

num_frames = numel(img_files);

init_target_sz = target_sz;
%set the feature ratio to the feature-cell size
featureRatio = params.t_global.cell_size;
search_area = prod(init_target_sz / featureRatio * search_area_scale);

% when the number of cells are small, choose a smaller cell size
if isfield(params.t_global, 'cell_selection_thresh')
    if search_area < params.t_global.cell_selection_thresh * filter_max_area
        params.t_global.cell_size = min(featureRatio, max(1, ceil(sqrt(prod(init_target_sz * search_area_scale)/(params.t_global.cell_selection_thresh * filter_max_area)))));
        
        featureRatio = params.t_global.cell_size;
        search_area = prod(init_target_sz / featureRatio * search_area_scale);
    end
end
global_feat_params = params.t_global;

if search_area > filter_max_area
    currentScaleFactor = sqrt(search_area / filter_max_area);
else
    currentScaleFactor = 1.0;
end
% target size at the initial scale
base_target_sz = target_sz/currentScaleFactor;
%window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        sz = floor( base_target_sz * search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        sz = repmat(sqrt(prod(base_target_sz * search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        sz = base_target_sz + sqrt(prod(base_target_sz * search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    otherwise
        error('Unknown "params.search_area_shape". Must be ''proportional'', ''square'' or ''fix_padding''');
end
% set the size to exactly match the cell size
sz = round(sz / featureRatio) * featureRatio;
use_sz = floor(sz/featureRatio);
% window size, taking padding into account
% sz = floor(base_target_sz * (1 + padding));

% desired translation filter output (gaussian shaped), bandwidth
% proportional to target size
% construct the label function
output_sigma = sqrt(prod(floor(base_target_sz/featureRatio))) * output_sigma_factor;
rg = circshift(-floor((use_sz(1)-1)/2):ceil((use_sz(1)-1)/2), [0 -floor((use_sz(1)-1)/2)]);
cg = circshift(-floor((use_sz(2)-1)/2):ceil((use_sz(2)-1)/2), [0 -floor((use_sz(2)-1)/2)]);
% [rs, cs] = ndgrid(-floor((use_sz(1) - 1)/2) : ceil((use_sz(1) - 1)/2), -floor((use_sz(2) - 1)/2) : ceil((use_sz(2) - 1)/2));
[rs, cs] = ndgrid( rg,cg);
y = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
yf = single(fft2(y));

if interpolate_response == 1
    interp_sz = use_sz * featureRatio;
else
    interp_sz = use_sz;
end

% desired scale filter output (gaussian shaped), bandwidth proportional to
% number of scales
% scale_sigma = nScales/sqrt(33) * scale_sigma_factor;
% ss = (1:nScales) - ceil(nScales/2);
% ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
% ysf = single(fft(ys));

% store pre-computed translation filter cosine window
cos_window = single(hann(use_sz(1)) * hann(use_sz(2))');
% the search area size
% support_sz = prod(use_sz);
% Calculate feature dimension
im = imread(img_files{1});
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        colorImage = false;
    else
        colorImage = true;
    end
else
    colorImage = false;
end
if size(im,3) > 1 && colorImage == false
    im = im(:,:,1);
end
% store pre-computed scale filter cosine window
% if mod(nScales,2) == 0
%     scale_window = single(hann(nScales+1));
%     scale_window = scale_window(2:end);
% else
%     scale_window = single(hann(nScales));
% end;
if params.use_reg_window
    % create weight window
    % normalization factor
    reg_scale = 0.5 * base_target_sz/featureRatio;
    % construct grid
    wrg = -(use_sz(1)-1)/2:(use_sz(1)-1)/2;
    wcg = -(use_sz(2)-1)/2:(use_sz(2)-1)/2;
    [wrs, wcs] = ndgrid(wrg, wcg);
    % construct the regukarization window
    reg_window = exp(-(abs(wrs/reg_scale(1)).^params.reg_window_power + abs(wcs/reg_scale(2)).^params.reg_window_power)/12);
end
% scale factors
if nScales > 0
    scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
    
    scaleFactors = scale_step .^ scale_exp;
    
    %force reasonable scale changes
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

% compute the resize dimensions used for feature extraction in the scale
% estimation
% scale_model_factor = 1;
% if prod(init_target_sz) > scale_model_max_area
%     scale_model_factor = sqrt(scale_model_max_area/prod(init_target_sz));
% end
% scale_model_sz = floor(init_target_sz * scale_model_factor);
if interpolate_response >= 3
    % Pre-computes the grid that is used for socre optimization
%     ky = -floor((use_sz(1) - 1)/2) : ceil((use_sz(1) - 1)/2);
%     kx = -floor((use_sz(2) - 1)/2): ceil((use_sz(2) - 1)/2);
%     kx = kx';
    ky = circshift(-floor((use_sz(1) - 1)/2) : ceil((use_sz(1) - 1)/2), [1, -floor((use_sz(1) - 1)/2)]);
    kx = circshift(-floor((use_sz(2) - 1)/2) : ceil((use_sz(2) - 1)/2), [1, -floor((use_sz(2) - 1)/2)])';
    newton_iterations = params.newton_iterations;
end
% currentScaleFactor = 1;

% to calculate precision
positions = zeros(numel(img_files), 4);

% to calculate FPS
time = 0;

% find maximum and minimum scales
% im = imread([video_path img_files{1}]);
% min_scale_factor = scale_step ^ ceil(log(max(5 ./ sz)) / log(scale_step));
% max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
multires_pixel_template = zeros(sz(1), sz(2), size(im,3), nScales, 'uint8');
for frame = 1:num_frames,
    %load image
    im = imread(img_files{frame});
    if size(im,3) > 1 && colorImage == false
        im = im(:,:,1);
    end
    tic;
%     if frame==184
%         system('pause');
%     end
    if frame > 1
        old_pos = inf(size(pos));
        iter = 1;
        %translation search
        while iter <= refinement_iterations && any(old_pos ~= pos)
            % Get multi-resolution image
            for scale_ind = 1:nScales
               multires_pixel_template(:,:,:,scale_ind) = ...
                   get_pixels(im, pos, round(sz*currentScaleFactor*scaleFactors(scale_ind)), sz);
            end
            xt = bsxfun(@times,get_features(multires_pixel_template,features,global_feat_params),cos_window);
%             xt = cellfun(@(feature_map) bsxfun(@times, feature_map, cos_window), get_features(multires_pixel_template,features,global_feat_params), 'uniformoutput', false);
            % calculate the correlation response of the translation filter
            xtf = fft2(xt);
%             xtf = cellfun(@(x) fft2(x),xt,'uniformoutput', false);
%             hf_m=cell2mat(hf_mode_alpha);
%             xtf_m=cell2mat(xtf);
%             responsef = permute(sum(bsxfun(@times,hf_m, xtf_m), 3),[1 2 4 3]);
%             responsef = cellfun(@(hf,x) permute(sum(bsxfun(@times,hf, x), 3),[1 2 4 3]),hf_mode_alpha,xtf,'uniformoutput', false);
%             responsef = sum(cat(4,responsef{:}),4);
            responsef = permute(sum(bsxfun(@times,hf_mode_alpha, xtf), 3),[1 2 4 3]);
             % if we undersampled features, we want to interpolate the
            % response so it has the same size as the image patch
            if interpolate_response == 2
                % use dynamic interp size
                interp_sz = floor(size(y) * featureRatio * currentScaleFactor);
            end
            responsef_padded = resizeDFT2(responsef, interp_sz);
            
            % response
            response = ifft2(responsef_padded, 'symmetric');
            % find maximum
            if interpolate_response == 3
                error('Invalid parameter value for interpolate_response');
            elseif interpolate_response == 4
                [disp_row, disp_col, sind] = resp_newton(response, responsef_padded, newton_iterations, ky, kx, use_sz);
            else
                [row, col, sind] = ind2sub(size(response), find(response == max(response(:)), 1));
                disp_row = mod(row - 1 + floor((interp_sz(1)-1)/2), interp_sz(1)) - floor((interp_sz(1)-1)/2);
                disp_col = mod(col - 1 + floor((interp_sz(2)-1)/2), interp_sz(2)) - floor((interp_sz(2)-1)/2);
            end
            % calculate translation
            switch interpolate_response
                case 0
                    translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor * scaleFactors(sind));
                case 1
                    translation_vec = round([disp_row, disp_col] * currentScaleFactor * scaleFactors(sind));
                case 2
                    translation_vec = round([disp_row, disp_col] * scaleFactors(sind));
                case 3
                    translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor * scaleFactors(sind));
                case 4
                    translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor * scaleFactors(sind));
            end
            % set the scale
            currentScaleFactor = currentScaleFactor * scaleFactors(sind);
            % adjust to make sure we are not to large or to small
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end
            
            % update position
            old_pos = pos;
            pos = pos + translation_vec;
            
            iter = iter + 1;
            %--------------------------------------------------------------------------------------------------------
            % calculate PSR value
            PSR=CalPSR(response(:,:,sind),11,use_sz);
        end
%         % extract the test sample feature map for the translation filter
%         xt = get_translation_sample(im, pos, sz, currentScaleFactor, cos_window);
%         
%         % calculate the correlation response of the translation filter
%         xtf = fft2(xt);
%         response = real(ifft2(sum(hf_mode_alpha .* xtf, 3)));
%         
%         % find the maximum translation response
%         [row, col] = find(response == max(response(:)), 1);
%         
%         % update the position
%         pos = pos + round((-sz/2 + [row, col]) * currentScaleFactor);
%         
%         % extract the test sample feature map for the scale filter
%         xs = get_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);
%         
%         % calculate the correlation response of the scale filter
%         xsf = fft(xs,[],2);
%         scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den + lambda)));
%         
%         % find the maximum scale response
%         recovered_scale = find(scale_response == max(scale_response(:)), 1);
%         
%         % update the scale
%         currentScaleFactor = currentScaleFactor * scaleFactors(recovered_scale);
%         if currentScaleFactor < min_scale_factor
%             currentScaleFactor = min_scale_factor;
%         elseif currentScaleFactor > max_scale_factor
%             currentScaleFactor = max_scale_factor;
%         end
    end
    
    % extract the training sample feature map for the translation filter
%     xl = get_translation_sample(im, pos, sz, currentScaleFactor, cos_window);
    appearance_pixels = get_pixels(im,pos,round(base_target_sz*currentScaleFactor),base_target_sz);
    
    if frame~=1
        kappa =0.05;
        appearance_diff = double(appearance_pixels(:))/255-double(appearance_pixels_old(:))/255;
        dist         = exp(-kappa*sum(appearance_diff.^2));
    else
        appearance_pixels_old = appearance_pixels;
    end
    pixels = get_pixels(im,pos,round(sz*currentScaleFactor),sz);
    % extract features and do windowing
    xl = bsxfun(@times,get_features(pixels,features,global_feat_params),cos_window);
%     xl = cellfun(@(feature_map) bsxfun(@times, feature_map, cos_window), get_features(pixels,features,global_feat_params), 'uniformoutput', false);
    % calculate the translation filter update
    xlf = fft2(xl);
    new_hf_num = bsxfun(@times, yf, conj(xlf));
    new_hf_den = sum(xlf .* conj(xlf), 3);
%     xlf = cellfun(@(x) fft2(x),xl,'uniformoutput', false);
%     new_hf_num = cellfun(@(x) bsxfun(@times, yf, conj(x)), xlf,'uniformoutput', false);
%     new_hf_den = cellfun(@(x) sum(x .* conj(x), 3), xlf,'uniformoutput', false);
    
%     % extract the training sample feature map for the scale filter
%     xs = get_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);
%     
%     % calculate the scale filter update
%     xsf = fft(xs,[],2);
%     new_sf_num = bsxfun(@times, ysf, conj(xsf));
%     new_sf_den = sum(xsf .* conj(xsf), 1);
    
    
    if frame == 1
        % first frame, train with a single image
        hf_den = new_hf_den;
        hf_num = new_hf_num;
        
%         sf_den = new_sf_den;
%         sf_num = new_sf_num;
    else
        % subsequent frames, update the model
        if PSR<20 && dist<0.2
            learning_rate = learning_rate*0.1;
        end
        hf_den = (1 - learning_rate) * hf_den + learning_rate * new_hf_den;
        hf_num = (1 - learning_rate) * hf_num + learning_rate * new_hf_num;
        appearance_pixels_old=(1 - learning_rate) * appearance_pixels_old + learning_rate * appearance_pixels;
%         sf_den = (1 - learning_rate) * sf_den + learning_rate * new_sf_den;
%         sf_num = (1 - learning_rate) * sf_num + learning_rate * new_sf_num;
    end
%     hf_mode_alpha=cellfun(@(hn,hd) bsxfun(@times,hn,1./(hd+lambda)),hf_num,hf_den,'uniformoutput', false);
%     figure(1),surf(double(real(ifft2(hf_mode_alpha(:,:,1)))));
    
%     h1=double(real(ifft2(hf_mode_alpha(:,:,1))));
%     iff_hf_mode_alpha=ifft2(hf_mode_alpha);
%     h_index=abs(iff_hf_mode_alpha)>4*1e-4;
%     iff_hf_mode_alpha =bsxfun(@times,iff_hf_mode_alpha,h_index);
%     hf_mode_alpha     =fft2(iff_hf_mode_alpha);
%     for ii=1:size(hf_mode_alpha,3)
%         figure(ii),surf(double(real(iff_hf_mode_alpha(:,:,ii))));
%     end


     [hf_mode_alpha]=SparseCorrelationFilterSolver(hf_den,hf_num,1-reg_window,5,100); %2 10   曾经的最优参数8和5
%     h2=double(real(ifft2(hf_mode_alpha(:,:,1))));
%     figure(2),surf(h2);
    
    % calculate the new target size
    target_sz = floor(base_target_sz * currentScaleFactor);
    
    %save position
    positions(frame,:) = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
    
    time = time + toc;
    
    
    %visualization
    if visualization ==1
        rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        if isToolboxAvailable('Computer Vision System Toolbox')
            im = insertShape(im, 'Rectangle', rect_position_vis, 'LineWidth', 4, 'Color', 'red');
%                 im = insertShape(im, 'Rectangle', rect_position_padded, 'LineWidth', 4, 'Color', 'yellow');
                % Display the annotated video frame using the video player object. 
            step(params.videoPlayer, im);
        end
    end
%     if visualization == 1
%         rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
%         im_to_show = double(im)/255;
%         if size(im_to_show,3) == 1
%             im_to_show = repmat(im_to_show, [1 1 3]);
%         end
%         if frame == 1
%             fig_handle = figure('Name', 'Tracking');
%             imagesc(im_to_show);
%             hold on;
%             rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
%             text(10, 10, int2str(frame), 'color', [0 1 1]);
%             hold off;
%             axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
%         else
%             resp_sz = round(sz*currentScaleFactor*scaleFactors(scale_ind));
%             xs = floor(old_pos(2)) + (1:resp_sz(2)) - floor(resp_sz(2)/2);
%             ys = floor(old_pos(1)) + (1:resp_sz(1)) - floor(resp_sz(1)/2);
%             sc_ind = floor((nScales - 1)/2) + 1;
%             
%             figure(fig_handle);
%             imagesc(im_to_show);
%             hold on;
%             resp_handle = imagesc(xs, ys, fftshift(response(:,:,sc_ind))); colormap hsv;
%             alpha(resp_handle, 0.5);
%             rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
%             text(10, 10, int2str(frame), 'color', [0 1 1]);
%             hold off;
%         end
% %         if frame == 1,  %first frame, create GUI
% %             figure('Number','off', 'Name',['Tracker - ' video_path]);
% %             im_handle = imshow(uint8(im), 'Border','tight', 'InitialMag', 100 + 100 * (length(im) < 500));
% %             rect_handle = rectangle('Position',rect_position, 'EdgeColor','g');
% %             text_handle = text(10, 10, int2str(frame));
% %             set(text_handle, 'color', [0 1 1]);
% %         else
% %             try  %subsequent frames, update GUI
% %                 set(im_handle, 'CData', im)
% %                 set(rect_handle, 'Position', rect_position)
% %                 set(text_handle, 'string', int2str(frame));
% %             catch
% %                 return
% %             end
% %         end
%         
%         drawnow
% %         pause
%     end
end

fps = num_frames/time;