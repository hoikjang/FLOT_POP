from config import parse_args
args = parse_args()
if args.prediction:
    ####################################################################################################
    ###### 패키지 Import
    ####################################################################################################
    import numpy as np
    import pickle
    import holoviews as hv
    from holoviews import opts
    import tensorflow as tf
    from datetime import datetime
    import gc
    import pandas as pd
    from keras import backend as K
    import time
    import pytz
    KST = pytz.timezone('Asia/Seoul')

    import psycopg2 
    import jaydebeapi as jp
    from sqlalchemy import create_engine, text, URL # SQLAlchemy 2.0 이상 설치

    # os.environ['KMP_DUPLICATE_LIB_OK']='True' 
    tf.keras.backend.clear_session()
    # tf.config.run_functions_eagerly(True)
    hv.extension('matplotlib')

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


    ####################################################################################################
    ###### 모델링 정의
    ####################################################################################################
    
    # 디렉터리 정보
    fol_pkl         = args.result_folder # 피클 저장 폴더 

    # DB 적재 여부
    prrslt_db_upld  = args.prrslt_db_upld

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

    time_split_dt     = args.time_split_dt
    num_of_output     = args.num_of_output #depth of predicted output map

    model_type          = f'kt_flt_pop_{YYYYMM}'
    model_type_ver      = args.model_type_ver   
    time_interval       = args.time_interval    
    period_num          = args.period_num       
    post_plot_process   = args.post_plot_process
    seed                = args.seed             
    bool_external       = args.bool_external 
    flt_pop_500         = args.flt_pop_500
    flt_pop_50          = args.flt_pop_50

    tf.random.set_seed(
        seed
    )


    df_timerange = pd.date_range(strt_YYYYMMDD, end_YYYYMMDD, freq='M')
    list_YYYYMM  = df_timerange.strftime("%Y%m").tolist()

    pstrt_YYYYMMDDHH = args.pstrt_YYYYMMDDHH
    pend_YYYYMMDDHH  = args.pend_YYYYMMDDHH 
    ptime_split_dt   = args.ptime_split_dt  
    bus_out_500      = args.bus_out_500     
    bus_db_input     = args.bus_db_input    
    
    rsltn            = '500'




    ### 유동인구 Inflow, Outflow 격자 데이터 Model Input Data 로드

    tbl_name     = flt_pop_500

    with open(f'{fol_pkl}/{tbl_name}_input_{strt_YYYYMM}_{end_YYYYMM}.pkl', 'rb') as f:
        np_base_input = pickle.load(f)


    # Model Structure Setting
    map_height                = np_base_input.shape[1]
    map_width                 = np_base_input.shape[2]
    middle_length             = np_base_input.shape[1]*np_base_input.shape[2]


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

    del np_base_input
    gc.collect()

    ## Hive 테이블
    Hive_DB       = args.Hive_DB 


    ## posgresql
    host               = args.host          
    port               = args.port          
    database           = args.database      
    username           = args.username      
    password           = args.password      
    driver_psgrsql     = args.driver_psgrsql
    dialect            = args.dialect       

    ## hive
    driver_hv    = args.driver_hv
    url          = args.url      
    hivejar      = args.hivejar  

    ####################################################################################################
    ###### DB 접속 엔진 형성 
    ####################################################################################################

    ## postgresql


    def connect_postgresql(host, database, username, password, port):
        conn_postgresql = psycopg2.connect(
            host     = host,
            dbname   = database,
            user     = username,
            password = password,
            port     = port
        )
        return conn_postgresql

    url_object = URL.create(
        drivername = f"{dialect}+{driver_psgrsql}",
        username   = username,
        password   = password,
        host       = host,
        database   = database,
        port       = port
    )
    engine                = create_engine(url=url_object)

    conn_postgresql = connect_postgresql(host, database, username, password, port)
    cur_postgresql  = conn_postgresql.cursor()

    ## hive
    conn_hive = jp.connect(driver_hv, url, ['nifiadmin', 'nifi135@$^'], hivejar)

    ####################################################################################################
    ###### 승하차 데이터 로드 및 Input 마트 형성
    ####################################################################################################


    df_bus                              = pd.read_pickle(f'/home/ggits/mls/01.유동인구_밀집도_예측/DATASET/{bus_out_500}.pkl')
    list_bus_cll                        = df_bus[f'cell_{rsltn}_id'].drop_duplicates()[:middle_length].to_list()



    if time_interval !='1hour':
        df_time = df_bus[['yyyymmddhh',f'time_idx']].drop_duplicates()
    else:
        df_time = df_bus[['yyyymmddhh','time_idx']].drop_duplicates()

    df_bus_sub                          = df_bus.loc[df_bus[f'cell_{rsltn}_id'].isin(list_bus_cll)].sort_values(by=['time_idx',f'cell_{rsltn}_id'])
    # 메모리 해제
    del df_bus
    gc.collect()

    df_bus_sub_sub                      = df_bus_sub[['time_idx','cell_500_id','pass_sum']].drop_duplicates()
    df_bus_sub_sub[f'cell_{rsltn}_idx'] = df_bus_sub_sub['cell_500_id'].factorize()[0]

    with open(f'{fol_pkl}/{bus_out_500}_input_base_{strt_YYYYMM}_{end_YYYYMM}.pkl', 'wb') as f:
        pickle.dump(df_bus_sub_sub, f)

    # 메모리 해제
    del df_bus_sub
    gc.collect()

    length                      = len(df_time['time_idx'].drop_duplicates())
    np_base_input_fltpop_light  = np.full((length, middle_length ,1),0)
    np_base_fltpop_light        = df_bus_sub_sub[['time_idx',f'cell_{rsltn}_idx',f'pass_sum']].drop_duplicates().to_numpy()

    for i in range(np_base_fltpop_light.shape[0]):
        time_idx1   = np_base_fltpop_light[i][0]
        middle_idx1 = np_base_fltpop_light[i][1]
        fltpop_val1 = np_base_fltpop_light[i][2]

        np_base_input_fltpop_light[int(time_idx1),int(middle_idx1),:] = fltpop_val1

    np_base_input = np.reshape(np_base_input_fltpop_light,(np_base_input_fltpop_light.shape[0],map_height,map_width,1))

    with open(f'{fol_pkl}/{bus_out_500}_input_{strt_YYYYMM}_{end_YYYYMM}.pkl', 'wb') as f:
        pickle.dump(np_base_input, f)

    del df_bus_sub_sub, np_base_fltpop_light, np_base_input_fltpop_light
    gc.collect()

    if time_interval !='1hour':
        time_start     = f'{pstrt_YYYYMMDDHH}00'
        time_end       = f'{pend_YYYYMMDDHH}00'
        time_start_idx = int(df_time.loc[(df_time['yyyymmddhh']==time_start)][f'time_idx'])
        time_end_idx   = int(df_time.loc[(df_time['yyyymmddhh']==time_end)][f'time_idx'])
    else:
        time_start     = pstrt_YYYYMMDDHH
        time_end       = pend_YYYYMMDDHH
        time_start_idx = int(df_time.loc[(df_time['yyyymmddhh']==time_start)][f'time_idx'])
        time_end_idx   = int(df_time.loc[(df_time['yyyymmddhh']==time_end)][f'time_idx'])

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

    ####################################################################################################
    ###### ST_ResNet 모델 학습, Validation 나누기
    ####################################################################################################
    if time_interval !='1hour':
        time_split     = f'{ptime_split_dt}00'
        time_split_idx = int(df_time.loc[(df_time['yyyymmddhh']==time_split)][f'time_idx'])
        split_num      = time_split_idx-time_start_idx
    else:
        time_split     = ptime_split_dt
        time_split_idx = int(df_time.loc[(df_time['yyyymmddhh']==time_split)][f'time_idx'])
        split_num      = time_split_idx-time_start_idx

    ####################################################################################################
    ###### ST_ResNet 모델 로드
    ####################################################################################################

    model = tf.keras.models.load_model(f'{fol_pkl}/stresnet_saved_models/{model_type}_{model_type_ver}')


    ####################################################################################################
    ###### ST_ResNet 모델 Test 데이터 예측 - 승하차
    ####################################################################################################
    if time_interval !='1hour':
        if bool_external:
            y_pred = model.predict([
                                        x_temporal[split_num:]
                                        ,x_closeness[split_num:]
                                        ,x_period[split_num:]
                                        ,x_trend[split_num:]
                                        ,x_external[split_num:]])  
            del x_temporal, x_closeness, x_period, x_trend, x_external
        else:
            y_pred = model.predict([
                                    x_temporal[split_num:]
                                    ,x_closeness[split_num:]
                                    ,x_period[split_num:]
                                    ,x_trend[split_num:]])

            del x_temporal, x_closeness, x_period, x_trend, x_external
    else:
        if bool_external:
            y_pred = model.predict([
                                        x_closeness[split_num:]
                                        ,x_period[split_num:]
                                        ,x_trend[split_num:]
                                        ,x_external[split_num:]])  
            del x_temporal, x_closeness, x_period, x_trend, x_external
        else:
            y_pred = model.predict([
                                        x_closeness[split_num:]
                                        ,x_period[split_num:]
                                        ,x_trend[split_num:]])  
                                        
            del x_closeness, x_period, x_trend
    


    # 메모리 해제

    gc.collect()

    ####################################################################################################
    ###### ST_ResNet 모델 Test(True), Prediction 데이터 저장
    ####################################################################################################
    with open(f'{fol_pkl}/y_pred_{model_type}_{model_type_ver}.pkl', 'wb') as f:
        pickle.dump(y_pred, f)


    with open(f'{fol_pkl}/y_true_{model_type}_{model_type_ver}.pkl', 'wb') as f:
        pickle.dump(y[split_num:,:,:,:], f)

    # 메모리 해제
    del y_pred, y
    gc.collect()


    ####################################################################################################
    ###### ST_ResNet 모델 Test, Prediction 데이터 로드
    ####################################################################################################

    time_interval       = '1hour'
    y_pred = pd.read_pickle(f'{fol_pkl}/y_pred_{model_type}_{model_type_ver}.pkl')
    y_true = pd.read_pickle(f'{fol_pkl}/y_true_{model_type}_{model_type_ver}.pkl')

    ####################################################################################################
    ###### Root Mean Squared Error 산출
    ####################################################################################################

    rmseloss = tf.keras.metrics.RootMeanSquaredError()(y_true[0,:,:,:], y_pred[0,:,:,:])
    np.mean(rmseloss)

    ####################################################################################################
    ###### Mean Absolute Error 산출
    ####################################################################################################

    maeloss = tf.keras.metrics.mean_absolute_error(y_true[0,:,:,:], y_pred[0,:,:,:])
    np.mean(maeloss)

    ####################################################################################################
    ###### Mean Absolute Percentage Error 산출
    ####################################################################################################

    mapeloss = tf.keras.metrics.mean_absolute_percentage_error(y_true[0,:,:,:], y_pred[0,:,:,:])
    np.mean(mapeloss)

    ####################################################################################################
    ###### Mean Squared Logarithmic Error 산출
    ####################################################################################################
    msleloss = tf.keras.losses.MeanSquaredLogarithmicError()(y_true[0,:,:,:], y_pred[0,:,:,:])
    np.mean(msleloss)

    ####################################################################################################
    ###### 예측값 마트 생성
    ####################################################################################################

    list_tmp      = []
    np_y_pred     = np.array(y_pred[0,:,:,:])
    np_y_pred_org = np_y_pred.reshape(middle_length,num_of_output)

    df_bus_input_base     = pd.read_pickle(f'{fol_pkl}/{bus_out_500}_input_base_{strt_YYYYMM}_{end_YYYYMM}.pkl')
    list_bus_input_cell   = df_bus_input_base[[f'cell_{rsltn}_id',f'cell_{rsltn}_idx']].drop_duplicates()[f'cell_{rsltn}_id'].to_list()
    list_baseyyyymmddhh   = df_time.loc[(df_time['time_idx']>time_split_idx)&(df_time['time_idx']<=time_split_idx+num_of_output)]['yyyymmddhh'].to_list()


    cell_len = np_y_pred_org.shape[0]
    time_len = np_y_pred_org.shape[1]
    dcrr_tm  = datetime.now(KST)
    etl_time = datetime.strftime(dcrr_tm, '%Y%m%d%H%M%S%f')[:-3]
    for idx1, cell_idx in enumerate(range(cell_len)):
        for idx2, time_idx in enumerate(range(time_len)):
            list_tmp.append([list_baseyyyymmddhh[time_idx][:8],list_baseyyyymmddhh[time_idx][-2:],list_bus_input_cell[cell_idx],np_y_pred_org[cell_idx,time_idx],etl_time])
    df_bus_output         = pd.DataFrame(list_tmp)
    df_bus_output.columns = ['baseymd','timezn','cell_500','flt_pop','etl_dt']

    with open(f'{fol_pkl}/bus_db_input_{model_type}_{model_type_ver}.pkl', 'wb') as f:
        pickle.dump(df_bus_output, f)

    del df_time, df_bus_input_base, list_tmp, list_bus_input_cell, list_baseyyyymmddhh
    gc.collect()

    ####################################################################################################
    ###### 예측값 적재
    ####################################################################################################

    if prrslt_db_upld:
        db_name     = database
        tbl_nm_inpt = bus_db_input
        YYYYMMDD    = ptime_split_dt[:8]

        dst_tm  = datetime.now(KST)
        print(f'''[LOG] {db_name} DB {tbl_nm_inpt}_{YYYYMMDD} 데이터 삭제 시작, 시작시간 = {datetime.strftime(dst_tm, '%Y%m%d %H:%M:%S')}''')

        ## 데이터 초기화

        conn_postgresql = connect_postgresql(host, database, username, password, port)
        cur_postgresql  = conn_postgresql.cursor()
        cur_postgresql.execute(f"""delete from {tbl_nm_inpt} where etl_dt ='{etl_time}'""")

        dend_tm = datetime.now(KST)
        del_tm  = dend_tm-dst_tm
        print(f'''[LOG] {db_name} DB {tbl_nm_inpt}_{YYYYMMDD} 데이터 삭제 마침, 마침시간 = {datetime.strftime(dend_tm, '%Y%m%d %H:%M:%S')}, 소요시간 = {del_tm}''')

        ## 데이터 로드
        dst_tm  = datetime.now(KST)
        print(f'''[LOG] {tbl_nm_inpt}_{YYYYMMDD} 마트 Import 시작, 시작시간 = {datetime.strftime(dst_tm, '%Y%m%d %H:%M:%S')}''')

        df_input      = pd.read_pickle(f'{fol_pkl}/bus_db_input_{model_type}_{model_type_ver}.pkl')


        dend_tm = datetime.now(KST)
        del_tm  = dend_tm-dst_tm
        print(f'''[LOG] {tbl_nm_inpt}_{YYYYMMDD} 마트 Import 마침, 마침시간 = {datetime.strftime(dend_tm, '%Y%m%d %H:%M:%S')}, 소요시간 = {del_tm}''')

        ## 데이터 적재
        list_tmp_pk = ['baseymd', 'timezn','cell_500']

        dst_tm        = datetime.now(KST)
        print(f'''[LOG] {db_name} DB {tbl_nm_inpt}_{YYYYMMDD} 데이터 적재 시작, 시작시간 = {datetime.strftime(dst_tm, '%Y%m%d %H:%M:%S')}''')


        df_input = df_input.astype({
                         'baseymd'       : object
                        ,'timezn'        : object
                        ,'cell_500'      : object
                        ,'flt_pop'       : np.float32
                        ,'etl_dt'        : object
                        }
        )

        df_input['flt_pop'] = np.round(df_input['flt_pop'],0)

        df_input.to_sql(
              name      = f'{tbl_nm_inpt}'
            , con       = engine
            , if_exists = 'append'
            , index     = False
            , schema    = database
            )

        dend_tm = datetime.now(KST)
        del_tm  = dend_tm-dst_tm
        print(f'''[LOG] {db_name} DB {tbl_nm_inpt}_{YYYYMMDD} 데이터 적재 마침, 마침시간 = {datetime.strftime(dend_tm, '%Y%m%d %H:%M:%S')}, 소요시간 = {del_tm}''')

        ##적재 건수 확인

        DF건수          = len(df_input.drop_duplicates(list_tmp_pk))
        conn_postgresql = connect_postgresql(host, database, username, password, port)

        DB건수 = pd.read_sql(
            f"""
                select count(1) as 건수
                from {tbl_nm_inpt}
                where etl_dt = '{etl_time}'
            """
        , conn_postgresql
        ).values[0][0]

        print(rf"[LOG] DF건수 = {DF건수:,}, DB건수 = {DB건수:,}")

        del df_input
        gc.collect()