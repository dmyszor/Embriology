function DataAnalysis_EvaluateNetworkDataSingleFileStep2()
    networkPath = 'PATH\ClassificationModels\SSRS models\1.mat';        
    sourceCSVDatasetFolder = 'PATH\SegmentationModel';

    numberOfCategoriesConst = 3;
    categoriesSetByUserOutput = 1:1:numberOfCategoriesConst; 
    categoricalCategoriesOutput = categorical(categoriesSetByUserOutput);
    classesOutput = categories(categoricalCategoriesOutput);
    
    [dsTrain] = PrepareTrainDataStore(categoriesSetByUserOutput, sourceCSVDatasetFolder);
    
    
    load(networkPath,'dlNet');     
    miniBatchValidationSize = 1;
    [output] = Validate(dlNet, dsTrain, miniBatchValidationSize, classesOutput);

    if output == '1'
        disp("Oocyte type MI")
    end
    if output == '2'
        disp("Oocyte type MII")
    end
    if output == '3'
        disp("Oocyte type PI")
    end    
end

function [dlYPredDecoded ] = Validate(dlNet, dsValidationTrain, miniBatchValidationSize, classesOutput)
    mbq = minibatchqueue(dsValidationTrain,...
        'MiniBatchSize',miniBatchValidationSize,...
        'MiniBatchFcn', @preprocessData,...
        'MiniBatchFormat',{'SSCB','CB'});
    while hasdata(mbq)
        [dlX, dl] = next(mbq);
        dlYPred = predict(dlNet, dlX, 'Outputs', "OutputLayerSoftMax");        
        dlYPredDecoded = onehotdecode(dlYPred, classesOutput, 1);
    end
end

function [X,Y] = preprocessData(XCell, cellClass)        
    X = cat(4,XCell{:});
    Y = cat(2,cellClass{:}); 
end

function [dsTrain] = PrepareTrainDataStore(categoriesSetByUser, folder)
    resolution = 590;
    layerAmount = 2;
    yLabelsFlat = zeros(1, 1);

    xImages = zeros(resolution, resolution, layerAmount, 1);

    gv = table2array(readtable(fullfile(folder,'_GV.csv')));
    fpb = table2array(readtable(fullfile(folder, '_FPB.csv')));
    imgExpanded = zeros(590,590,2);
    imgExpanded(40:512+40-1,40:512+40-1,1) = gv;
    imgExpanded(40:512+40-1,40:512+40-1,2) = fpb;
    xImages (:, :, :, 1) = imgExpanded(:, :, :);
    dsXTrain = arrayDatastore(xImages(:,:,:,1:1:1), 'IterationDimension', 4);
    yLabelsFlat(1) = 0;
    yLabels = onehotencode(yLabelsFlat, 1, 'ClassNames', categoriesSetByUser);
    dsYTrain = arrayDatastore(yLabels(:, 1:1:1), 'IterationDimension', 2);
    dsTrain = combine (dsXTrain, dsYTrain);
end



