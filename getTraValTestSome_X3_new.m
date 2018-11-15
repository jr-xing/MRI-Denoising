%{
1. generate index array
2. load data
3. split data, channel processing, append to valGt, etc.
                                  csSubj**                
+------------------------------------------------------------------------+
|           |---96---|                                                   |
|           |--------|                                                   |
|           |--------|                                                   |
|           |-|bin2|-|                                                   |
|           |--------|                                                   |
|           |--------|                                                   |
+------------------------------------------------------------------------+
|                                                                        |
|<----------------------------------960--------------------------------->|
|                                                                        |

%}

%% Set path and other parameters
% workPath = '/media/remussn/Document/xdocument/development/mri/workSpace';
% dataPath = '/export/project/xiaojianxu/jiarui/MRI/workSpace/data_seg';
% savePath = '/export/project/xiaojianxu/jiarui/MRI/workSpace'
%dataPath = '/export/project/jiarui.xing/Projects/MRI/workSpace/data_seg';
%savePath = '/export/project/jiarui.xing/Projects/MRI/workSpace';

dataPath = '/export/project/xiaojianxu/jiarui/MRI/workSpace/data/data_seg_1000';
savePath = '/export/project/xiaojianxu/jiarui/MRI/workSpace/data';

traGtFiles = {'csSubj01','csSubj02','csSubj03','csSubj04','csSubj05','csSubj06','csSubj07','csSubj09'};    
traObFiles = {'mcSubj01','mcSubj02','mcSubj03','mcSubj04','mcSubj05','mcSubj06','mcSubj07','mcSubj09'};
valGtFiles = {'csSubj08'};               valObFiles = {'mcSubj08'};
numSlice = 960; 
binLength = 96; binNum = numSlice/binLength;
% binSliceStart = 25; binSliceEnd = 75; 
binSliceStart = 1; binSliceEnd = 96; 
indArr = [];
for binIdx = 1:binNum
    indArr = horzcat(indArr, binSliceStart + (binIdx-1)*binLength:binSliceEnd+(binIdx-1)*binLength);
end


%% Load, process and save imgs
addChan = true;
addChanMode = 'neigh_motion';
% addChanMode = 'neigh_motion_time';
%traGt = load_split_save('traGt', traGtFiles, workPath, addChan, addChanMode, indArr);
%traOb = load_split_save('traOb', traObFiles, workPath, addChan, addChanMode, indArr);
%valGt = load_split_save('valGt', valGtFiles, workPath, addChan, addChanMode, 1:960);
%valOb = load_split_save('valOb', valObFiles, workPath, addChan, addChanMode, 1:960);
%load_split_save('traGt', traGtFiles, dataPath, savePath, addChan, addChanMode, indArr, 3, 'FULL_SEG');
%load_split_save('traGt', traGtFiles, dataPath, savePath, false, addChanMode, indArr, 3, 'FULL_SEG');
%load_split_save('traOb', traObFiles, dataPath, savePath, false,   addChanMode, indArr, 3, 'FULL_SEG');
%load_split_save('traOb', traObFiles, dataPath, savePath, addChan, addChanMode, indArr, 3, 'FULL_SEG');
%load_split_save('valGt', valGtFiles, workPath, false, addChanMode, 100:5:245, 1, 'FULL');
%load_split_save('valOb', valObFiles, workPath, false, addChanMode, 100:5:245, 1, 'FULL');

%load_split_save('traOb', traObFiles, dataPath, savePath, false,   addChanMode, indArr, 3, 'FULL_SEG_1000');
%load_split_save('traOb', traObFiles, dataPath, savePath, addChan, addChanMode, indArr, 3, 'FULL_SEG_1000');
%load_split_save('valOb', valObFiles, dataPath, savePath, addChan, addChanMode, 100:5:245,1,'FULL_SEG_1000');
 load_split_save('valOb', valObFiles, dataPath, savePath, addChan, addChanMode, indArr, 1, 'FULL_SEG_1000_FULL');
%load_split_save('valOb', valObFiles, workPath, addChan, addChanMode, 100:5:245, 1, 'FULL');

%% Utility functions
function imgsMC = someAddChannel(imgs, mode, indArr)
% Extract part of imgs according to indArr 
% and add gray imgs to "color" imgs by adding neighbour slices
% imgs: [w,h,N] matrix
% indArr: 
% mode: copy / neigh_time / neigh_motion
% imgsMC: [w,h,3,N] matrix - MultiChannel imgs
    if nargin == 2
        indArr = 1:size(imgs,3);  
    end
    imgsFullNum = size(imgs, 3);
    indArrLen = size(indArr, 2);    
    if strcmp(mode, 'copy')
        imgsMC = zeros(size(imgs,1),size(imgs,2),3,indArrLen);
        imgsMC(:,:,1,:) = imgs(:,:,indArr);
        imgsMC(:,:,2,:) = imgs(:,:,indArr);
        imgsMC(:,:,3,:) = imgs(:,:,indArr);
    elseif strcmp(mode, 'neigh_time')
        imgsMC = zeros(size(imgs,1),size(imgs,2),3,indArrLen);
        imgsMC(:,:,1,:) = imgs(:,:,max(indArr-1, 1));
        imgsMC(:,:,2,:) = imgs(:,:,indArr);
        imgsMC(:,:,3,:) = imgs(:,:,min(indArr+1, indArrLen));        
    elseif strcmp(mode, 'neigh_motion')
        imgsMC = zeros(size(imgs,1),size(imgs,2),3,indArrLen);
        binLength = 96;
        prevBinIdx = ((indArr-binLength)<=0).*indArr + ((indArr-binLength)>0).*(indArr-binLength);
        nextBinIdx = ((indArr+binLength)>imgsFullNum).*indArr + ((indArr+binLength)<=imgsFullNum).*(indArr+binLength);
        imgsMC(:,:,1,:) = imgs(:,:,prevBinIdx);
        imgsMC(:,:,2,:) = imgs(:,:,indArr);
        imgsMC(:,:,3,:) = imgs(:,:,nextBinIdx);
    elseif strcmp(mode, 'neigh_motion_time')
        imgsMC = zeros(size(imgs,1),size(imgs,2),5,indArrLen);
        binLength = 96;
        prevBinIdx = ((indArr-binLength)<=0).*indArr + ((indArr-binLength)>0).*(indArr-binLength);
        nextBinIdx = ((indArr+binLength)>imgsFullNum).*indArr + ((indArr+binLength)<=imgsFullNum).*(indArr+binLength);
        imgsMC(:,:,1,:) = imgs(:,:,max(indArr-1, 1));
        imgsMC(:,:,2,:) = imgs(:,:,prevBinIdx);
        imgsMC(:,:,3,:) = imgs(:,:,indArr);
        imgsMC(:,:,4,:) = imgs(:,:,nextBinIdx);
        imgsMC(:,:,5,:) = imgs(:,:,min(indArr+1, indArrLen));
    end

end

function currentVar = loadMRIMat0(filePath)
    currentFile = load(filePath);
    currentFile = struct2cell(currentFile);
    currentVar = currentFile{1};
end


% function imgs = load_split_save(fileType, fileNames, workPath, addChan, addChanMode, indArr)
function load_split_save(fileType, fileNames, dataPath, savePath, addChan, addChanMode, indArr, splitNum, filename_prefix)
    if nargin == 7
        filename_prefix = '';
    else
        filename_prefix = ['_' filename_prefix];
    end
    disp(['Processing ' fileType '...']);tic
    fileNumEachPart = ceil(size(fileNames,2)/splitNum);    
    for splitIdx = 1:splitNum        
        fileStartIdx = (splitIdx-1)*fileNumEachPart + 1;
        fileEndIdx = min(splitIdx*fileNumEachPart, size(fileNames,2));
        disp(['Processing Part ' int2str(splitIdx) '/' int2str(splitNum) '(File ' int2str(fileStartIdx) ' to ' int2str(fileEndIdx) ')...'])
        for traFileIdx = fileStartIdx:fileEndIdx
            % load and process file, then append to imgs
            disp(['Processing File ' int2str(traFileIdx) '/' int2str(size(fileNames,2)) '...'])
            currentVar = single(loadMRIMat0([dataPath '/' fileNames{traFileIdx} '.mat']));

            % add channels if addChan
            if addChan
                currentVar = someAddChannel(currentVar, addChanMode, indArr);
            else
                currentVar = currentVar(:,:,indArr);
            end

            % concatenate imgs in files
            disp('concatenating...')
            if exist('imgs', 'var')                
                if addChan
                    imgs = cat(4, imgs, currentVar);
                else
                    imgs = cat(3, imgs, currentVar);
                end
            else
                imgs = currentVar; 
            end
        end
        clear currentVar;

        % Save
        disp('Saving...')
        % Setting Saving Path
        if strcmp(fileType, 'traGt') || strcmp(fileType, 'traOb')
            saveSubPath = '/train_np/';
        elseif strcmp(fileType, 'valGt') || strcmp(fileType, 'valOb')
            saveSubPath = '/valid_np/';
        end

        % Saving
        % dimension format for tensorflow: [N, w, h, c]
        % change to single to save storage    
        if splitNum == 1
            if addChan
                imgs_tf = permute(single(imgs), [4,1,2,3]);
                fileName = [savePath saveSubPath fileType filename_prefix '_' addChanMode '.mat'];                
            else
                imgs_tf = permute(single(imgs), [3,1,2]);
                fileName = [savePath saveSubPath fileType filename_prefix '.mat']
            end
        else            
            if addChan
                imgs_tf = permute(single(imgs), [4,1,2,3]);
                fileName = [savePath saveSubPath fileType filename_prefix '_' addChanMode '_part_' int2str(splitIdx) '.mat'];
                % save(fileName, 'imgs_tf','-v7.3');
            else
                imgs_tf = permute(single(imgs), [3,1,2]);
                fileName = [savePath saveSubPath fileType filename_prefix '_part_' int2str(splitIdx) '.mat'];
            end
        end
        save(fileName, 'imgs_tf','-v7.3');
        disp(['Finish Saving' fileName]);toc
        clear imgs;
    end
    disp('Finish Saving!');toc
end