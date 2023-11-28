from config import parse_args
args = parse_args()
if args.etl_2:
    ####################################################################################################
    ###### 패키지 Import
    ####################################################################################################
    import numpy as np
    import pickle
    import gc
    import jaydebeapi as jp
    import pandas as pd
    from datetime import datetime
    import psycopg2 
    import pytz
    KST = pytz.timezone('Asia/Seoul')

    from datetime import datetime
    from sqlalchemy import create_engine, text, URL # SQLAlchemy 2.0 이상 설치
    # SQLAlchemy 1.x 버전에서는 URL 메서드 활용 불가


    ####################################################################################################
    ###### 모델 기본 정보
    ####################################################################################################
    

    # 디렉터리 정보
    fol_pkl       = args.result_folder # 피클 저장 폴더 
    ## 데이터 기본 정보
    # 데이터 정보
    region        = args.region
    if args.region:
        rsltn = '50'
    else:
        rsltn = '500'      
    YYYYMM        = args.YYYYMM
    strt_YYYYMM   = args.strt_YYYYMM
    end_YYYYMM    = args.end_YYYYMM

    strt_YYYYMMDD = args.strt_YYYYMMDD
    end_YYYYMMDD  = args.end_YYYYMMDD # EX) 6월까지 원하는 경우 그 다음달 1일, 20230-07-01 입력


    df_timerange = pd.date_range(strt_YYYYMMDD, end_YYYYMMDD, freq='M')
    list_YYYYMM  = df_timerange.strftime("%Y%m").tolist()

    # 격자 생성 컷오프 기준
    thrshld          =  args.thrshld
    if rsltn == '50':
        area_size = 2500.0
    else:
        area_size = 250000.0
    
    ## postgresql 테이블
    postgresql_DB   = args.postgresql_DB  
    tbl_cll_mppng   = args.tbl_cll_mppng  

    ## Hive 테이블
    Hive_DB       = args.Hive_DB 
    tbl_dly       = args.tbl_dly  
    tbl_hrly      = args.tbl_hrly
    flt_pop_      = args.flt_pop_

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
    # conn_hive = jp.connect(driver_hv, url, ['nifiadmin', 'nifi135@$^'], hivejar)

    ####################################################################################################
    ###### Model Input Mart - Light 생성 함수 
    ####################################################################################################

    def prime_num(value):
        factors  = []
        list_tmp = []
        for i in range(1, int(value**0.5)+1):
            if value % i == 0:
                val_1 = i
                val_2 = value / i
                factors.append((val_1, val_2))
                list_tmp.append(abs(val_1-val_2))

        return list_tmp.index(min(list_tmp))+1, factors[list_tmp.index(min(list_tmp))]
    ####################################################################################################
    ###### Model Input Mart Base 생성 
    ####################################################################################################
     
    tbl_nm_inpt   = flt_pop_ + rsltn
    df_base       = pd.DataFrame()
    tbl_name      = tbl_dly


    dst_tm      = datetime.now(KST)
    print(f'''[LOG] {tbl_nm_inpt}_{YYYYMM} Input Mart Base 생성 시작, 시작시간 = {datetime.strftime(dst_tm, '%Y%m%d %H:%M:%S')}''')

    df_base_flt = pd.DataFrame()
    for idx1, YYYYMM in enumerate(list_YYYYMM):
        df_tmp      = pd.read_pickle(f'{fol_pkl}/{tbl_nm_inpt}_{YYYYMM}.pkl')
        df_tmp      = df_tmp[['기준년월','요일','시간대','셀코드','유동인구수']].drop_duplicates() #수정 -> 유동인구수
        df_base_flt = pd.concat([df_base_flt,df_tmp])
        del df_tmp
        gc.collect()

    if df_base_flt['셀코드'].isna().any():
        df_base_flt = df_base_flt.loc[~df_base_flt['셀코드'].isna()]

    df_base_flt['셀코드'] = df_base_flt[f'셀코드'].astype(np.uint32)
    df_base               = df_base.merge(df_base_flt)
    
    del df_base_flt
    gc.collect() 

    df_base['기준시간대']     = df_base['기준년월'].astype(str)+ df_base['요일'].astype(str) + df_base['시간대'].astype(str).str.zfill(2)   
    df_base['시간대인덱스']   = df_base['기준시간대'].factorize()[0]
    df_base['유동인구수']     = df_base['유동인구수'].fillna(0)


    with open(f'{fol_pkl}/{tbl_nm_inpt}_input_mart_base_{strt_YYYYMM}_{end_YYYYMM}.pkl','wb') as f:
        pickle.dump(df_base,f)

    del df_base
    gc.collect()

    dend_tm = datetime.now(KST)
    del_tm  = dend_tm-dst_tm
    print(f'''[LOG] {tbl_nm_inpt}_{YYYYMM} Input Mart Base 생성 마침, 마침시간 = {datetime.strftime(dend_tm, '%Y%m%d %H:%M:%S')}, 소요시간 = {del_tm}''')


    ####################################################################################################
    ###### Model Input Mart - Light 생성 
    ####################################################################################################
    tbl_name        = flt_pop_ + rsltn
    dim             =  0.001
    

    dst_tm      = datetime.now(KST)
    print(f'''[LOG] {tbl_nm_inpt}_{YYYYMM} Input Mart Light 생성 시작, 시작시간 = {datetime.strftime(dst_tm, '%Y%m%d %H:%M:%S')}''')

    df_base         = pd.read_pickle(f'{fol_pkl}/{tbl_name}_input_mart_base_{strt_YYYYMM}_{end_YYYYMM}.pkl')
    df_base_sub     = df_base[['시간대인덱스','셀코드','X축','Y축','유동인구수']].drop_duplicates()

    del df_base
    gc.collect()


    df_base_sub['셀코드_유동인구수_평균'] = df_base_sub.groupby('셀코드')['유동인구수'].transform('mean')
    df_base_stt                            = df_base_sub[['셀코드','셀코드_유동인구수_평균']].drop_duplicates()


    for val in np.arange(0,1,dim):
        thrshld_val = df_base_stt['셀코드_유동인구수_평균'].quantile(val)
        if (thrshld_val/area_size > thrshld):
            break

    del df_base_stt
    gc.collect()

    df_base_sub                                  = df_base_sub.loc[df_base_sub['셀코드_유동인구수_평균']>=thrshld_val].drop_duplicates().sort_values(by=f'셀코드')
    df_base_sub[f'셀코드_cidx']                  = df_base_sub['셀코드'].factorize()[0]
    list_base_cll                                = df_base_sub['셀코드'].drop_duplicates().to_list() 
    df_base_cll_list                             = df_base_sub[['셀코드','셀코드_cidx','X축','Y축']].drop_duplicates()

    with open(f'{fol_pkl}/{tbl_name}_cll_list_{strt_YYYYMM}_{end_YYYYMM}.pkl', 'wb') as f:
        pickle.dump(df_base_cll_list, f)

    del df_base_cll_list
    gc.collect()

    df_base_sub_sub      = df_base_sub.loc[df_base_sub[f'셀코드'].isin(list_base_cll)].drop_duplicates()

    with open(f'{fol_pkl}/{tbl_name}_sub_{strt_YYYYMM}_{end_YYYYMM}.pkl', 'wb') as f:
        pickle.dump(df_base_sub, f)

    del df_base_sub
    gc.collect()

    length        = len(df_base_sub_sub['시간대인덱스'].drop_duplicates())
    middle_length = len(list_base_cll)

    num, factors  = prime_num(middle_length)
    if num == 1:
        middle_length +=1
        num, factors = prime_num(middle_length)



    np_base_input_fltpop_light  = np.full((length, middle_length ,1),0)
    np_base_fltpop_light        = df_base_sub_sub[['시간대인덱스','셀코드_cidx','유동인구수']].drop_duplicates().to_numpy()

    for i in range(np_base_fltpop_light.shape[0]):
        time_idx1   = np_base_fltpop_light[i][0]
        middle_idx1 = np_base_fltpop_light[i][1]
        fltpop_val1 = np_base_fltpop_light[i][2]

        np_base_input_fltpop_light[int(time_idx1),int(middle_idx1),:] = fltpop_val1

    with open(f'{fol_pkl}/{tbl_name}_input_mart_{strt_YYYYMM}_{end_YYYYMM}.pkl', 'wb') as f:
        pickle.dump(np.reshape(np_base_input_fltpop_light,(np_base_input_fltpop_light.shape[0],int(factors[0]),int(factors[1]),1)), f)


    # # 메모리 해제
    del df_base_sub_sub ,np_base_input_fltpop_light, np_base_fltpop_light
    gc.collect()

    dend_tm = datetime.now(KST)
    del_tm  = dend_tm-dst_tm
    print(f'''[LOG] {tbl_nm_inpt}_{YYYYMM} Input Mart Light 생성 마침, 마침시간 = {datetime.strftime(dend_tm, '%Y%m%d %H:%M:%S')}, 소요시간 = {del_tm}''')

