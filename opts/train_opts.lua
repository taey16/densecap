local M = { }

function M.parse(arg)

  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Train a DenseCap model.')
  cmd:text()
  cmd:text('Options')

  local data_h5 = 
    '/data2/visualgenome/regions_descriptions.json.h5'
  local data_json = 
    '/data2/visualgenome/regions_descriptions.json.json'
  local dataset_name = 
    'visualgenome'

  local torch_model = 
    '/data2/ImageNet/ILSVRC2012/torch_cache/X_gpu1_resception_nag_lr0.00450_decay_start0_every160000/model_29.t7'

  local sampler_batch_size = 256
  local sampler_high_thresh = 0.7
  local sampler_low_thresh = 0.3
  local rpn_hidden_dim = 512

  local mid_box_reg_weight = 0.05
  local mid_objectness_weight = 0.1
  local end_box_reg_weight = 0.1
  local end_objectness_weight = 0.1
  local captioning_weight = 1.0

  local drop_prob = 0.5
  local rnn_size = 512
  local input_encoding_size = 512

  local finetune_cnn_after = 0
  local learning_rate = 1e-5
  local cnn_learning_rate = 1e-5
  local cnn_optim = 'adam'
  local learning_rate_decay_seed = 
    0.9
    -- -1
  local learning_rate_decay_start = 
    77396 * 5
  local learning_rate_decay_every = 
    77396
  local weight_decay = 1e-6
  local box_reg_decay = 5e-5

  local eval_first_iteration = 1
  local test_interval = 20000

  local retrain_iter = 
    500000
    --0
  local checkpoint_start_from =
    '/storage/visualgenome/checkpoints/vgg16_lr0.000010_cnnlr0.000010_seed0.900000_start386980_every77396_wc0.000001_finetune0_iter0/checkpoints.t7'
    --''
  local checkpoint_path = string.format(
    '/storage/%s/checkpoints/vgg16_lr%f_%s_cnnlr%f_seed%f_start%d_every%d_wc%f_finetune%d_iter%d/', 
    dataset_name, 
    learning_rate, cnn_optim, cnn_learning_rate,
    learning_rate_decay_seed, learning_rate_decay_start, learning_rate_decay_every, 
    weight_decay, finetune_cnn_after, retrain_iter
  )


  if checkpoint_start_from ~= '' and retrain_iter == 0 then
    print(string.format('retrain from %s', start_from))
    error(string.format('retrain_iter MUST NOT be zero'))
  end

  -- Core ConvNet settings
  cmd:option('-backend', 'cudnn', 'nn|cudnn')

  -- Model settings
  cmd:option('-rpn_hidden_dim', rpn_hidden_dim,
    'Hidden size for the extra convolution in the RPN')
  cmd:option('-sampler_batch_size', sampler_batch_size,
    'Batch size to use in the box sampler')
  cmd:option('-rnn_size', rnn_size,
    'Number of units to use at each layer of the RNN')
  cmd:option('-input_encoding_size', input_encoding_size,
    'Dimension of the word vectors to use in the RNN')
  cmd:option('-sampler_high_thresh', sampler_high_thresh,
    'Boxes with IoU greater than this with a GT box are considered positives')
  cmd:option('-sampler_low_thresh', sampler_low_thresh,
    'Boxes with IoU less than this with all GT boxes are considered negatives')
  cmd:option('-train_remove_outbounds_boxes', 1,
    'Whether to ignore out-of-bounds boxes for sampling at training time')
  
  -- Loss function weights
  cmd:option('-mid_box_reg_weight', mid_box_reg_weight,
    'Weight for box regression in the RPN')
  cmd:option('-mid_objectness_weight', mid_objectness_weight,
    'Weight for box classification in the RPN')
  cmd:option('-end_box_reg_weight', end_box_reg_weight,
    'Weight for box regression in the recognition network')
  cmd:option('-end_objectness_weight', end_objectness_weight,
    'Weight for box classification in the recognition network')
  cmd:option('-captioning_weight', captioning_weight, 'Weight for captioning loss')
  cmd:option('-weight_decay', weight_decay, 'L2 weight decay penalty strength')
  cmd:option('-box_reg_decay', box_reg_decay,
    'Strength of pull that boxes experience towards their anchor')

  -- Data input settings
  cmd:option('-data_h5', data_h5,
    'HDF5 file containing the preprocessed dataset (from proprocess.py)')
  cmd:option('-data_json', data_json,
    'JSON file containing additional dataset info (from preprocess.py)')
  cmd:option('-proposal_regions_h5', '',
    'override RPN boxes with boxes from this h5 file (empty = don\'t override)')
  cmd:option('-debug_max_train_images', -1,
    'Use this many training images (for debugging); -1 to use all images')

  -- Optimization
  cmd:option('-learning_rate', learning_rate, 'learning rate to use')
  cmd:option('-cnn_learning_rate', cnn_learning_rate, 'learning rate to use')
  cmd:option('-learning_rate_decay_seed', learning_rate_decay_seed,
    'decay_factor = math.pow(opt.learning_rate_decay_seed, frac)')
  cmd:option('-learning_rate_decay_start', learning_rate_decay_start, 
    'at what iteration to start decaying learning rate? (-1 = dont)')
  cmd:option('-learning_rate_decay_every', learning_rate_decay_every, 
    'every how many iterations thereafter to drop LR by half?')
  cmd:option('-optim_beta1', 0.9, 'beta1 for adam')
  cmd:option('-optim_beta2', 0.999, 'beta2 for adam')
  cmd:option('-optim_epsilon', 1e-8, 'epsilon for smoothing')
  cmd:option('-cnn_optim', cnn_optim, 'optimization to use for CNN')
  cmd:option('-cnn_optim_alpha',0.9, 'alpha for momentum of CNN')
  cmd:option('-cnn_optim_beta',0.999, 'beta for momentum of CNN')
  cmd:option('-drop_prob', drop_prob, 'Dropout strength throughout the model.')
  cmd:option('-max_iters', -1, 'Number of iterations to run; -1 to run forever')
  cmd:option('-retrain_iter', retrain_iter, 'starting iter for retrain')
  cmd:option('-checkpoint_start_from', checkpoint_start_from,
    'Load model from a checkpoint instead of random initialization.')
  cmd:option('-finetune_cnn_after', finetune_cnn_after,
    'Start finetuning CNN after this many iterations (-1 = never finetune)')
  cmd:option('-val_images_use', -1, 'Number of validation images to use for evaluation; -1 to use all')

  -- Model checkpointing
  cmd:option('-save_checkpoint_every', test_interval, 'How often to save model checkpoints')
  cmd:option('-checkpoint_path', checkpoint_path, 'Name of the checkpoint file to use')

  -- Test-time model options (for evaluation)
  cmd:option('-test_rpn_nms_thresh', 0.7,
    'Test-time NMS threshold to use in the RPN')
  cmd:option('-test_final_nms_thresh', 0.3,
    'Test-time NMS threshold to use for final outputs')
  cmd:option('-test_num_proposals', 1000,
    'Number of region proposal to use at test-time')

  -- Visualization
  cmd:option('-progress_dump_every', test_interval,
    'Every how many iterations do we write a progress report to vis/out ?. 0 = disable.')
  cmd:option('-losses_log_every', 0,
    'How often do we save losses, for inclusion in the progress dump? (0 = disable)')

  -- Misc
  cmd:option('-id', '', 'an id identifying this run/job; useful for cross-validation')
  cmd:option('-seed', 123, 'random number generator seed to use')
  cmd:option('-gpu', 0, 'which gpu to use. -1 = use CPU')
  cmd:option('-timing', false, 'whether to time parts of the net')
  cmd:option('-clip_final_boxes', 1, 'Whether to clip final boxes to image boundar')
  cmd:option('-eval_first_iteration', eval_first_iteration, 
    'evaluate on first iteration? 1 = do, 0 = dont.')
  cmd:option('-display', 5, 'display interval')

  print('checkpoint_path: ' .. checkpoint_path)
  io.flush()

  cmd:text()
  local opt = cmd:parse(arg or {})
  return opt
end

return M
