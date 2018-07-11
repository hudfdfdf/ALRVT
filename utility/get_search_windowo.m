function window_sz = get_search_windowo( target_sz, im_sz, padding)
% GET_SEARCH_WINDOW
% if sqrt(prod(target_sz))/sqrt(prod(im_sz(1:2)))<0.08  %0.08
% %     for smaller target.      e.g.  Skiing
%     target_sz = floor(target_sz.*1.3);
% end
if(target_sz(1)/target_sz(2) > 2)
    % For objects with large height, we restrict the search window with padding.height
    window_sz = floor(target_sz.*[1+padding.height, 1+padding.generic]);%target_sz(1)/target_sz(2)  ]);
    
elseif(prod(target_sz)/prod(im_sz(1:2)) > 0.05)
    % For objects with large height and width and accounting for at least 10 percent of the whole image,
    % we only search 2x height and width
    window_sz=floor(target_sz*(1+padding.large));
% elseif sqrt(prod(target_sz))>85 && target_sz(1)/target_sz(2)>1.5
%     window_sz=floor(target_sz*(1+padding.large*0.6));
else
    %otherwise, we use the padding configuration
    window_sz = floor(target_sz * (1 + padding.generic));
    
end
% if target_sz(1)<20 || target_sz(2)<20
%     %for smaller target.      e.g.  bike
%     target_sz = floor(target_sz.*1.2);
%     window_sz = floor(target_sz * (1 + 4));
% end


end

