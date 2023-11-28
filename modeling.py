from config import parse_args
args = parse_args()
if args.modeling:
    ####################################################################################################
    ###### 패키지 Import
    ####################################################################################################
    import numpy as np
    import pickle
    import tensorflow as tf
    from datetime import datetime
    import gc
    import pandas as pd
    from keras import backend as K
    import time
    import pytz
    KST = pytz.timezone('Asia/Seoul')

    # os.environ['KMP_DUPLICATE_LIB_OK']='True' 
    tf.keras.backend.clear_session()
    # tf.config.run_functions_eagerly(True)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    ####################################################################################################
    ###### 모델링 정의
    ####################################################################################################

    
    # 디렉터리 정보
    fol_pkl         = args.result_folder # 피클 저장 폴더 


    ## 데이터 기간 정보
    YYYYMM        = args.YYYYMM
    strt_YYYYMM   = args.strt_YYYYMM
    end_YYYYMM    = args.end_YYYYMM

    strt_YYYYMMDD = args.strt_YYYYMMDD
    end_YYYYMMDD  = args.end_YYYYMMDD # EX) 6월까지 원하는 경우 그 다음달 1일, 20230-07-01 입력


    df_timerange = pd.date_range(strt_YYYYMMDD, end_YYYYMMDD, freq='M')
    list_YYYYMM  = df_timerange.strftime("%Y%m").tolist()


    strt_YYYYMMDDWHH  = f'{strt_YYYYMM}000'
    end_YYYYMMDDWHH   = f'{end_YYYYMM}623'

    date_1 = f'{strt_YYYYMMDD}  00:00:00'
    date_2 = f'{end_YYYYMMDD}  00:00:00'

    time_split_dt     = '202303623'
    num_of_output     = 168 #depth of predicted output map

    model_type          = f'kt_flt_pop_{YYYYMM}'
    model_type_ver      = args.model_type_ver   
    time_interval       = args.time_interval    
    period_num          = args.period_num       
    post_plot_process   = args.post_plot_process
    seed                = args.seed             
    bool_external       = args.bool_external 
    flt_pop_            = args.flt_pop_

    tf.random.set_seed(
        seed
    )


    df_timerange = pd.date_range(strt_YYYYMMDD, end_YYYYMMDD, freq='M')
    list_YYYYMM  = df_timerange.strftime("%Y%m").tolist()




    ### 유동인구 Inflow, Outflow 격자 데이터 Model Input Data 로드

    region        = args.region
    if args.region:
        rsltn = '50'
    else:
        rsltn = '500'      

    tbl_nm_inpt     = flt_pop_+ rsltn

    with open(f'{fol_pkl}/{tbl_nm_inpt}_input_mart_{strt_YYYYMM}_{end_YYYYMM}.pkl', 'rb') as f:
        np_base_input = pickle.load(f)
    # np_base_input.shape

    # Model Structure Setting
    map_height                = np_base_input.shape[1]
    map_width                 = np_base_input.shape[2]


    if time_interval !='1hour':
        temporal_sequence_length  = 4
    else:
        temporal_sequence_length  = 0
    closeness_sequence_length = int(num_of_output*2)
    period_sequence_length    = 14
    trend_sequence_length     = 3
    if bool_external == True:
        num_of_ext_features       = 4
    else:
        num_of_ext_features       = 0

    ####################################################################################################
    ###### 모듈 정의
    ####################################################################################################

    # Model Training Setting
    batch_size                = 32
    lr                        = 0.001
    num_epochs                = 1000
    patience                  = 20
    validation_split          = 0.2


    num_of_filters            = 64
    num_of_residual_units     = 12

    # Normalization Setting
    epsilon                   = 1e-7

    # Convolution Layer Setting
    kernel_size               = (3,3)

    # Model Structure 
    logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    def ResInput(inputs, filters, kernel_size,strides=(1,1), name = None):
        '''
        Defines the first (input) layer of the ResNet architecture
        '''
        outputs = tf.keras.layers.Conv2D(filters=filters, 
                                        kernel_size=kernel_size, 
                                        strides=strides,
                                        padding = "same", 
                                        name    = name + "conv_input")(inputs) 
        return outputs

    def ResUnit(inputs, filters, kernel_size, strides=(1,1), name = None):   
        '''
        Defines a residual unit
        input -> [layernorm->relu->conv] X 2 -> reslink -> output
        '''
        
        # use layernorm before applying convolution
        outputs = tf.keras.layers.BatchNormalization(epsilon = epsilon,
                                                    axis    = -1,
                                                    name    = name + 'layernorm1')(inputs)
        # apply relu activation
        outputs = tf.keras.layers.Activation("relu", name = name + 'ReLU1')(outputs)
        # perform a 2D convolution
        outputs = tf.keras.layers.Conv2D(filters=filters, 
                                        kernel_size=kernel_size, 
                                        strides=strides, 
                                        padding = "same",
                                        name    = name + "conv1")(outputs)         
        # use layernorm before applying convolution
        outputs = tf.keras.layers.BatchNormalization(epsilon = epsilon,
                                                    axis    = -1,
                                                    name    = name + 'layernorm2')(inputs)
        # relu activation
        outputs = tf.keras.layers.Activation("relu", name = name + 'ReLU2')(outputs)
        # perform a 2D convolution
        outputs = tf.keras.layers.Conv2D(filters=filters, 
                                        kernel_size=kernel_size, 
                                        strides=strides, 
                                        padding = "same",
                                        name    = name + "conv2")(outputs)                            
        # add a residual link        
        outputs = tf.keras.layers.Add(name = name + "add")([outputs,inputs])
        
        # (Additional Change) add relu activation
        # outputs = tf.keras.layers.Activation("relu", name=name + "_out")(outputs)
        return outputs    

    def ResNet(inputs, filters, kernel_size, repeats, name = None):
        '''
        Defines the ResNet architecture
        '''
        #apply repeats number of residual layers
        for layer_id in range(repeats):
            inputs = ResUnit(inputs, filters, kernel_size, strides=(1,1), name = name + str(layer_id))
        # (Additional Change)
        # outputs = inputs
        outputs = tf.keras.layers.Activation("relu", name = name + 'ReLU_end')(inputs)
        return outputs


    def ResInput(inputs, filters, kernel_size,strides=(1,1), name = None):
        '''
        Defines the first (input) layer of the ResNet architecture
        '''
        outputs = tf.keras.layers.Conv2D(filters=filters, 
                                        kernel_size=kernel_size, 
                                        strides=strides,
                                        padding = "same", 
                                        name    = name + "conv_input")(inputs) 
        return outputs

    def ResOutput(inputs, filters, kernel_size,strides=(1,1), name =None):
        '''
        Defines the last (output) layer of the ResNet architecture
        '''
        #applying the final convolution to the tec map with depth 1 (num of filters=1)
        outputs = tf.keras.layers.Conv2D(filters=filters, 
                                        kernel_size=kernel_size, 
                                        strides=strides,
                                        padding = "same", 
                                        name    = name + "conv_output")(inputs) 
        return outputs
    def ExtUnit(inputs, filters, num_of_output,kernel_size,strides=(1,1), name = None):
        '''
        Defines the first (input) layer of the ResNet architecture
        '''
        # use layernorm before applying convolution
        outputs = tf.keras.layers.BatchNormalization(epsilon = epsilon,
                                                    axis    = -1,
                                                    name    = name + 'layernorm1')(inputs)
        # apply relu activation
        outputs = tf.keras.layers.Activation("relu", name = name + 'ReLU1')(inputs)
        # perform a 2D convolution
        outputs = tf.keras.layers.Conv2D(filters=filters, 
                                        kernel_size=kernel_size, 
                                        strides=strides, 
                                        padding = "same",
                                        name    = name + "conv1")(inputs)   
        outputs = tf.keras.layers.Activation("relu", name = name + 'relu1')(outputs)
        outputs = tf.keras.layers.Conv2D(filters=filters, 
                                        kernel_size=kernel_size, 
                                        strides=strides, 
                                        padding = "same",
                                        name    = name + "conv2")(outputs)         
        # use layernorm before applying convolution
        outputs = tf.keras.layers.BatchNormalization(epsilon = epsilon,
                                                    axis    = -1,
                                                    name    = name + 'layernorm2')(inputs)
        # relu activation
        outputs = tf.keras.layers.Activation("relu", name = name + 'relu2')(outputs)
        # perform a 2D convolution
        outputs = tf.keras.layers.Conv2D(filters=num_of_output, 
                                        kernel_size=kernel_size, 
                                        strides=strides, 
                                        padding = "same",
                                        name    = name + "conv3")(outputs)    
        return outputs
        
    def Fusion(temporal_output, closeness_output, period_output, trend_output,F,num_of_output, name =None):
        '''
        Combining the output from the module into one tec map
        '''
        if time_interval !='1hour':            
            layer_concat = tf.keras.layers.Concatenate()([temporal_output, closeness_output, period_output, trend_output])
        else:
            layer_concat = tf.keras.layers.Concatenate()([closeness_output, period_output, trend_output])
        
        # perform a 2D convolution
        outputs = tf.keras.layers.Conv2D(filters=F, 
                                        kernel_size=kernel_size, 
                                        strides=(1,1), 
                                        padding = "same",
                                        name    = name + "conv1")(layer_concat)         
        # relu activation
        outputs = tf.keras.layers.Activation("relu", name = name + 'ReLU2')(outputs)
        # perform a 2D convolution
        outputs = tf.keras.layers.Conv2D(filters=F, 
                                        kernel_size=kernel_size, 
                                        strides=(1,1), 
                                        padding = "same",
                                        name    = name + "conv2")(outputs)  
        # relu activation
        outputs = tf.keras.layers.Activation("relu", name = name + 'ReLU3')(outputs)
        # perform a 2D convolution
        outputs = tf.keras.layers.Conv2D(filters=num_of_output, 
                                        kernel_size=kernel_size, 
                                        strides=(1,1), 
                                        padding = "same",
                                        name    = name + "conv3")(outputs)  
        return outputs    

    class ST_ResNet(object):
        def __init__(self):
            B  = batch_size
            H  = map_height                             
            W  = map_width 
            if time_interval !='1hour':
                M  = temporal_sequence_length
            C  = closeness_sequence_length
            P  = period_sequence_length
            T  = trend_sequence_length
            E  = num_of_ext_features
            O  = num_of_output
            F  = num_of_filters 
            U  = num_of_residual_units
            K  = kernel_size
            if time_interval !='1hour':
                self.temporal_input   = tf.keras.Input(dtype = tf.float32, shape=[H, W, M],  name="temporal")
            self.closeness_input  = tf.keras.Input(dtype = tf.float32, shape=[H, W, C],  name="closeness")
            self.period_input     = tf.keras.Input(dtype = tf.float32, shape=[H, W, P],  name="period")
            self.trend_input      = tf.keras.Input(dtype = tf.float32, shape=[H, W, T],  name="trend")
            self.external_input   = tf.keras.Input(dtype = tf.float32, shape=[H, W, E],  name="tmp")
                                            
            # ResNet architecture for the three modules
            # module 0: capturing temporal (recent)
            if time_interval !='1hour':
                self.temporal_output  = ResInput(inputs=self.temporal_input, filters=F, kernel_size=K, name = "temporal_input")
                self.temporal_output  = ResNet(inputs=self.temporal_output, filters=F, kernel_size=K, repeats=U, name = "resnet_temporal")
                self.temporal_output  = ResOutput(inputs=self.temporal_output, filters=1, kernel_size=K,name = "temporal_output")   
            # module 1: capturing closeness (recent)
            self.closeness_output = ResInput(inputs=self.closeness_input, filters=F, kernel_size=K, name = "closeness_input")
            self.closeness_output = ResNet(inputs=self.closeness_output, filters=F, kernel_size=K, repeats=U, name = "resnet_closeness")
            self.closeness_output = ResOutput(inputs=self.closeness_output, filters=1, kernel_size=K,name = "closeness_output")            
            # module 2: capturing period (near)
            self.period_output    = ResInput(inputs=self.period_input, filters=F, kernel_size=K,name = "period_input")
            self.period_output    = ResNet(inputs=self.period_output, filters=F, kernel_size=K, repeats=U, name = "resnet_period")
            self.period_output    = ResOutput(inputs=self.period_output, filters=1, kernel_size=K,name = "period_output")            
            # module 3: capturing trend (distant) 
            self.trend_output     = ResInput(inputs=self.trend_input, filters=F, kernel_size=K,name = "trend_input")
            self.trend_output     = ResNet(inputs=self.trend_output, filters=F, kernel_size=K, repeats=U,name = "resnet_trend")
            self.trend_output     = ResOutput(inputs=self.trend_output, filters=1, kernel_size=K,name = "trend_output")            
            # # parameter matrix based fusion
            if time_interval !='1hour':
                self.x_res            = Fusion(self.temporal_output, self.closeness_output, self.period_output, self.trend_output,F,num_of_output=O,name="fusion")
            else:
                self.x_res            = Fusion(0, self.closeness_output, self.period_output, self.trend_output,F,num_of_output=O,name="fusion")
            # # external values
            if bool_external:
                self.external_output  = ExtUnit(self.external_input, F ,O, kernel_size, strides=(1,1), name = "external")
            # # add res and external
                self.combine_output   = tf.keras.layers.Add(name = "final_addition")([self.x_res ,self.external_output])
                self.combine_output   = tf.keras.layers.Activation("relu", name ='relu_end')(self.combine_output)
            
            # # model generation
            if time_interval !='1hour':
                if bool_external:
                    self.model            = tf.keras.Model([self.temporal_input, self.closeness_input,self.period_input,self.trend_input,self.external_input],self.combine_output)
                else:
                    self.combine_output   = tf.keras.layers.Activation("relu", name ='relu_end')(self.x_res)
                    self.model            = tf.keras.Model([self.temporal_input, self.closeness_input,self.period_input,self.trend_input],self.combine_output)
            else:
                if bool_external:
                    self.model            = tf.keras.Model([self.closeness_input,self.period_input,self.trend_input,self.external_input],self.combine_output)
                else:
                    self.combine_output   = tf.keras.layers.Activation("relu", name ='relu_end')(self.x_res)
                    self.model            = tf.keras.Model([self.closeness_input,self.period_input,self.trend_input],self.combine_output)
            # #model compile
            self.model.compile(optimizer='adam', loss = 'mean_squared_error', metrics = [tf.keras.metrics.RootMeanSquaredError()])

    ####################################################################################################
    ###### Time 데이터 추출
    ####################################################################################################
    with open(f'{fol_pkl}/{tbl_nm_inpt}_input_mart_base_{strt_YYYYMM}_{end_YYYYMM}.pkl', 'rb') as f:
        df_base_mart = pickle.load(f)

        df_time = df_base_mart[['기준시간대','시간대인덱스']].drop_duplicates()


    # 메모리 해제
    del df_base_mart
    gc.collect()

    ####################################################################################################
    ###### External 데이터 Model Input Mart 로드
    ####################################################################################################
    if bool_external:
        dst_tm      = datetime.now(KST)
        print(f'''[LOG] {tbl_nm_inpt}_{YYYYMM} Input Mart External 로드 시작, 시작시간 = {datetime.strftime(dst_tm, '%Y%m%d %H:%M:%S')}''')
        with open(f'{fol_pkl}/base_external_{model_type}.pkl', 'rb') as f:
            df_external = pickle.load(f)
        dend_tm = datetime.now(KST)
        del_tm  = dend_tm1-dst_tm1
        print(f'''[LOG] {tbl_nm_inpt}_{YYYYMM} Input Mart External 로드 마침, 마침시간 = {datetime.strftime(dend_tm, '%Y%m%d %H:%M:%S')}, 소요시간 = {del_tm}''')

    ####################################################################################################
    ###### 유동인구 격자 데이터 Time_Step 확인
    ####################################################################################################

    date_format_str = '%Y-%m-%d %H:%M:%S'
    start = datetime.strptime(date_1, date_format_str)
    end   = datetime.strptime(date_2, date_format_str)
    # Get interval between two timstamps as timedelta object
    diff = end - start
    # Get interval between two timstamps in hours
    diff_in_time_step1 = diff.total_seconds() / (60*period_num)
    print(f'{model_type} 데이터 전체 기간 Time Step 확인',diff_in_time_step1)

    ####################################################################################################
    ###### 유동인구 격자 데이터 전체 데이터 Input 형성
    ####################################################################################################
    dst_tm      = datetime.now(KST)
    print(f'''[LOG] {tbl_nm_inpt}_{YYYYMM} Input Mart 학습데이터 형성 시작, 시작시간 = {datetime.strftime(dst_tm, '%Y%m%d %H:%M:%S')}''')

    if time_interval !='1hour':
        time_start     = f'{strt_YYYYMMDDWHH}00'
        time_end       = f'{end_YYYYMMDDWHH}00'
        time_start_idx = int(df_time.loc[(df_time['datetime_min']==time_start)][f'time_step_{time_interval}'])
        time_end_idx   = int(df_time.loc[(df_time['datetime_min']==time_end)][f'time_step_{time_interval}'])
    else:
        time_start     = strt_YYYYMMDDWHH
        time_end       = end_YYYYMMDDWHH
        time_start_idx = int(df_time.loc[(df_time['기준시간대']==time_start)][f'시간대인덱스'])
        time_end_idx   = int(df_time.loc[(df_time['기준시간대']==time_end)][f'시간대인덱스'])

    range1         = time_end_idx - time_start_idx +1


    if time_interval !='1hour':
        x_temporal     = np.full((range1, map_height, map_width, temporal_sequence_length),0)   

    x_closeness    = np.full((range1, map_height, map_width, closeness_sequence_length),0)
    x_period       = np.full((range1, map_height, map_width, period_sequence_length),0)
    x_trend        = np.full((range1, map_height, map_width, trend_sequence_length),0)
    y              = np.full((range1, map_height, map_width, num_of_output),0)

    if bool_external:
        np_external    = df_external.to_numpy()
        x_external     = np.full((range1, map_height, map_width, num_of_ext_features),0)


    m_num = 1
    c_num = int(1*60/period_num)
    p_num = int(24*60/period_num)
    t_num = int(24*7*60/period_num)

    for time_idx, num_hours in enumerate(np.arange(range1-num_of_output)):
        if time_interval !='1hour':
            for idx1 in range(temporal_sequence_length):
                x_temporal[time_idx,:,:,idx1]         = np_base_input[time_start_idx+time_idx-m_num*(idx1+1) ,:,:,0]

        for idx2 in range(closeness_sequence_length):
            x_closeness[time_idx,:,:,idx2]        = np_base_input[time_start_idx+time_idx-c_num*(idx2+1) ,:,:,0]
        
        for idx3 in range(period_sequence_length):
            x_period[time_idx,:,:,idx3]           = np_base_input[time_start_idx+time_idx-p_num*(idx3+1) ,:,:,0]
        
        for idx4 in range(trend_sequence_length):
            x_trend[time_idx,:,:,idx4]            = np_base_input[time_start_idx+time_idx-t_num*(idx4+1) ,:,:,0]
        
        for idx5 in range(num_of_output):
            y[time_idx,:,:,idx5]                  = np_base_input[time_start_idx+time_idx+idx5           ,:,:,0]
        if bool_external:
            for idx6 in range(num_of_ext_features):
                x_external[time_idx,:,:,idx6]         = np_external[time_start_idx+time_idx,idx6+1]
        
    # 메모리 해제
    if bool_external:
        del df_external, np_external, np_base_input
    else:
        del np_base_input
    gc.collect()

    dend_tm = datetime.now(KST)
    del_tm  = dend_tm1-dst_tm1
    print(f'''[LOG] {tbl_nm_inpt}_{YYYYMM} Input Mart 학습데이터 형성 마침, 마침시간 = {datetime.strftime(dend_tm, '%Y%m%d %H:%M:%S')}, 소요시간 = {del_tm}''')
    ####################################################################################################
    ###### ST_ResNet 모델 로드 및 모델 설정
    ####################################################################################################
    dst_tm      = datetime.now(KST)
    print(f'''[LOG] {tbl_nm_inpt}_{YYYYMM} 모델 로드 및 설정 시작, 시작시간 = {datetime.strftime(dst_tm, '%Y%m%d %H:%M:%S')}''')

    stresnet = ST_ResNet()
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    earlystopping = tf.keras.callbacks.EarlyStopping(
        monitor  = 'val_loss',
        patience = patience,
        restore_best_weights=True,
    )
    dend_tm = datetime.now(KST)
    del_tm  = dend_tm1-dst_tm1
    print(f'''[LOG] {tbl_nm_inpt}_{YYYYMM} 모델 로드 및 설정 마침, 마침시간 = {datetime.strftime(dend_tm, '%Y%m%d %H:%M:%S')}, 소요시간 = {del_tm}''')

    ####################################################################################################
    ###### ST_ResNet 모델 학습, Validation 나누기
    ####################################################################################################
    dst_tm      = datetime.now(KST)
    print(f'''[LOG] {tbl_nm_inpt}_{YYYYMM} 모델 Train, Test Split 구간 설정 시작, 시작시간 = {datetime.strftime(dst_tm, '%Y%m%d %H:%M:%S')}''')

    if time_interval !='1hour':
        time_split     = f'{time_split_dt}00'
        time_split_idx = int(df_time.loc[(df_time['기준시간대']==time_split)][f'시간대인덱스'])
        split_num      = time_split_idx-time_start_idx
    else:
        time_split     = time_split_dt
        time_split_idx = int(df_time.loc[(df_time['기준시간대']==time_split)][f'시간대인덱스'])
        split_num      = time_split_idx-time_start_idx

    # 메모리 해제
    del df_time
    gc.collect()

    dend_tm = datetime.now(KST)
    del_tm  = dend_tm1-dst_tm1
    print(f'''[LOG] {tbl_nm_inpt}_{YYYYMM} 모델 Train, Test Split 구간 설정 마침, 마침시간 = {datetime.strftime(dend_tm, '%Y%m%d %H:%M:%S')}, 소요시간 = {del_tm}''')

    ####################################################################################################
    ###### ST_ResNet 모델 학습
    ####################################################################################################
    dst_tm      = datetime.now(KST)
    print(f'''[LOG] {tbl_nm_inpt}_{YYYYMM} 모델 Train, Validation 시작, 시작시간 = {datetime.strftime(dst_tm, '%Y%m%d %H:%M:%S')}''')

    start_train_time = time.time()
    if time_interval !='1hour':
        if bool_external:
            stresnet.model.fit([ x_temporal[:split_num]
                                ,x_closeness[:split_num]
                                ,x_period[:split_num]
                                ,x_trend[:split_num]
                                ,x_external[:split_num]]
                                ,y[:split_num]
                                ,batch_size=batch_size
                                ,epochs=num_epochs
                                ,validation_split=validation_split
                                ,callbacks=[earlystopping])
        else:
            stresnet.model.fit([ x_temporal[:split_num]
                                ,x_closeness[:split_num]
                                ,x_period[:split_num]
                                ,x_trend[:split_num]
                                ,y[:split_num]]
                                ,batch_size=batch_size
                                ,epochs=num_epochs
                                ,validation_split=validation_split
                                ,callbacks=[earlystopping])
    else:
        if bool_external:
            stresnet.model.fit([
                            x_closeness[:split_num]
                            ,x_period[:split_num]
                            ,x_trend[:split_num]
                            ,x_external[:split_num]]
                            ,y[:split_num]
                            ,batch_size=batch_size
                            ,epochs=num_epochs
                            ,validation_split=validation_split
                            ,callbacks=[earlystopping])
        else:
            stresnet.model.fit([
                            x_closeness[:split_num]
                            ,x_period[:split_num]
                            ,x_trend[:split_num]]
                            ,y[:split_num]
                            ,batch_size=batch_size
                            ,epochs=num_epochs
                            ,validation_split=validation_split
                            ,callbacks=[earlystopping])
    end_train_time = time.time()

    dend_tm = datetime.now(KST)
    del_tm  = dend_tm1-dst_tm1
    print(f'''[LOG] {tbl_nm_inpt}_{YYYYMM} 모델 Train, Validation 마침, 마침시간 = {datetime.strftime(dend_tm, '%Y%m%d %H:%M:%S')}, 소요시간 = {del_tm}''')
    ###################################################################################################
    ##### ST_ResNet 모델 저장 및 저장 모델 확인
    ###################################################################################################

    dst_tm      = datetime.now(KST)
    print(f'''[LOG] {tbl_nm_inpt}_{YYYYMM} 모델 저장 시작, 시작시간 = {datetime.strftime(dst_tm, '%Y%m%d %H:%M:%S')}''')
    stresnet.model.save(f'{fol_pkl}/stresnet_saved_models/{model_type}_{model_type_ver}')

    new_model = tf.keras.models.load_model(f'{fol_pkl}/stresnet_saved_models/{model_type}_{model_type_ver}')

    # Check its architecture
    new_model.summary()
                            
    # 메모리 해제
    del stresnet, new_model
    gc.collect()

    dend_tm = datetime.now(KST)
    del_tm  = dend_tm1-dst_tm1
    print(f'''[LOG] {tbl_nm_inpt}_{YYYYMM} 모델 저장 마침, 마침시간 = {datetime.strftime(dend_tm, '%Y%m%d %H:%M:%S')}, 소요시간 = {del_tm}''')