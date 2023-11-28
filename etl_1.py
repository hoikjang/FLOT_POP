from config import parse_args
args = parse_args()
if args.etl_1:
    ####################################################################################################
    ###### 패키지 Import
    ####################################################################################################
    import numpy as np
    import pickle
    import gc
    import pandas as pd
    from datetime import datetime
    import pytz
    KST = pytz.timezone('Asia/Seoul')
    import psycopg2 
    import jaydebeapi as jp
    from sqlalchemy import create_engine, text, URL # SQLAlchemy 2.0 이상 설치
    # SQLAlchemy 1.x 버전에서는 URL 메서드 활용 불가

    ####################################################################################################
    ###### 모델 기본 정보
    ####################################################################################################
    ## 데이터 기간 정보


    # 디렉터리 정보
    base_dir        = args.base_dir
    fol_pkl         = args.result_folder # 피클 저장 폴더 
    
    # DB 쿼리 여부
    db_data_imprt   = args.db_data_imprt

    # DB 적재 여부
    inptmrt_db_upld = args.inptmrt_db_upld
    
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
    ###### 유동인구 시간대(hourly) 데이터 변환 - Pandas(데이터프레임)
    ####################################################################################################

    df_ctycd_ctynm    = pd.read_csv(f'{base_dir}/ctycd_ctynm.csv')
    if rsltn == '50':
        cty_cd_val        = df_ctycd_ctynm.loc[df_ctycd_ctynm['CTY_NM']==f'{region}']['CTY_CD'].values[0]
    sql_cll_mppng_rgn = f'''
        select 
            distinct
            cell_50_id
        from {tbl_cll_mppng} T1
        where city_cd = '{cty_cd_val}'
    '''


    sql_cll_mppng_500 = f'''
        select
            distinct
            cell_500_id
            ,cell_50_id
        from {tbl_cll_mppng}       
    '''

    list_sq = ['hourly','daily']
    for idx1, sq in enumerate(list_sq):
        for idx2, YYYYMM in enumerate(list_YYYYMM):
            if sq == 'hourly':
                tbl_name     = tbl_hrly
                convert_dict_1 = { 
                             'etl_ym'       : np.uint32
                            ,'cell_id'      : np.uint32
                            ,'x_axis'       : np.uint32
                            ,'y_axis'       : np.uint32
                            ,'tz_0'         : np.float32
                            ,'tz_1'         : np.float32
                            ,'tz_2'         : np.float32
                            ,'tz_3'         : np.float32
                            ,'tz_4'         : np.float32
                            ,'tz_5'         : np.float32
                            ,'tz_6'         : np.float32
                            ,'tz_7'         : np.float32
                            ,'tz_8'         : np.float32
                            ,'tz_9'         : np.float32
                            ,'tz_10'        : np.float32
                            ,'tz_11'        : np.float32
                            ,'tz_12'        : np.float32
                            ,'tz_13'        : np.float32
                            ,'tz_14'        : np.float32
                            ,'tz_15'        : np.float32
                            ,'tz_16'        : np.float32
                            ,'tz_17'        : np.float32
                            ,'tz_18'        : np.float32
                            ,'tz_19'        : np.float32
                            ,'tz_20'        : np.float32
                            ,'tz_21'        : np.float32
                            ,'tz_22'        : np.float32
                            ,'tz_23'        : np.float32
                }
                convert_dict_1_a = { 
                             'tz_0'         : '00'
                            ,'tz_1'         : '01'
                            ,'tz_2'         : '02'
                            ,'tz_3'         : '03'
                            ,'tz_4'         : '04'
                            ,'tz_5'         : '05'
                            ,'tz_6'         : '06'
                            ,'tz_7'         : '07'
                            ,'tz_8'         : '08'
                            ,'tz_9'         : '09'
                            ,'tz_10'        : '10'
                            ,'tz_11'        : '11'
                            ,'tz_12'        : '12'
                            ,'tz_13'        : '13'
                            ,'tz_14'        : '14'
                            ,'tz_15'        : '15'
                            ,'tz_16'        : '16'
                            ,'tz_17'        : '17'
                            ,'tz_18'        : '18'
                            ,'tz_19'        : '19'
                            ,'tz_20'        : '20'
                            ,'tz_21'        : '21'
                            ,'tz_22'        : '22'
                            ,'tz_23'        : '23'
                }
                convert_dict_2 = { 
                             'hours'        : np.uint8
                            ,'hourly_count' : np.float32
                }
                kor_dict = {
                             "etl_ym"          : '기준년월' 
                            ,"cell_id"         : '셀코드'
                            ,"x_axis"          : 'X축'
                            ,"y_axis"          : 'Y축'
                            ,"hours"           : '시간대'
                            ,"hourly_count"    : '시간대집계값'
                }
            else:
                tbl_name      = tbl_dly
                convert_dict_1 = { 
                         'etl_ym'        : np.uint32
                        ,'cell_id'       : np.uint32
                        ,'x_axis'        : np.uint32
                        ,'y_axis'        : np.uint32
                        ,'monday'        : np.float32
                        ,'tuesday'       : np.float32
                        ,'wednesday'     : np.float32
                        ,'thursday'      : np.float32
                        ,'friday'        : np.float32
                        ,'saturday'      : np.float32
                        ,'sunday'        : np.float32
                }
                convert_dict_1_a = { 
                             'monday'     : 'Monday'
                            ,'tuesday'    : 'Tuesday'    
                            ,'wednesday'  : 'Wednesday'
                            ,'thursday'   : 'Thursday'
                            ,'friday'     : 'Friday'
                            ,'saturday'   : 'Saturday'
                            ,'sunday'     : 'Sunday'
                }
                convert_dict_2 = { 
                         'dayofweek'        : object
                        ,'daily_count'      : np.float32
                }
                kor_dict = {
                         "etl_ym"          : '기준년월' 
                        ,"cell_id"         : '셀코드'
                        ,"x_axis"          : 'X축'
                        ,"y_axis"          : 'Y축'
                        ,"dayofweek"       : '요일별'
                        ,"daily_count"     : '요일별집계값'
                }
                

            ## 데이터 쿼리
            if db_data_imprt:
                dst_tm       = datetime.now(KST)
                print(f'''[LOG] {tbl_name}_{rsltn}_{YYYYMM} 쿼리 시작, 시작시간 = {datetime.strftime(dst_tm, '%Y%m%d %H:%M:%S')}''') 


                
                if sq == 'hourly':
                    if rsltn == '50':
                        sql_hrly_base  = f'''
                            select
                                etl_ym
                                ,cell_id         
                                ,x_axis          
                                ,y_axis          
                                ,tz_0      
                                ,tz_1       
                                ,tz_2      
                                ,tz_3       
                                ,tz_4      
                                ,tz_5      
                                ,tz_6       
                                ,tz_7       
                                ,tz_8     
                                ,tz_9       
                                ,tz_10     
                                ,tz_11      
                                ,tz_12      
                                ,tz_13     
                                ,tz_14      
                                ,tz_15      
                                ,tz_16     
                                ,tz_17      
                                ,tz_18      
                                ,tz_19      
                                ,tz_20      
                                ,tz_21      
                                ,tz_22      
                                ,tz_23     
                            from {tbl_hrly} T1
                            where     etl_ym = '{YYYYMM}'
                                and cell_id in (
                                    {sql_cll_mppng_rgn}
                                )
                            order by etl_ym
                                    ,cell_id
                        '''
                    if rsltn == '500':
                        sql_hrly_base  = f'''
                            select
                                T1.etl_ym                     as etl_ym
                                ,T2.cell_500_id                as cell_id
                                ,avg(T1.x_axis)::numeric(6)    as x_axis      
                                ,avg(T1.y_axis)::numeric(6)    as y_axis       
                                ,sum(T1.tz_0)                  as tz_0
                                ,sum(T1.tz_1)                  as tz_1
                                ,sum(T1.tz_2)                  as tz_2
                                ,sum(T1.tz_3)                  as tz_3
                                ,sum(T1.tz_4)                  as tz_4
                                ,sum(T1.tz_5)                  as tz_5
                                ,sum(T1.tz_6)                  as tz_6
                                ,sum(T1.tz_7)                  as tz_7
                                ,sum(T1.tz_8)                  as tz_8
                                ,sum(T1.tz_9)                  as tz_9
                                ,sum(T1.tz_10)                 as tz_10
                                ,sum(T1.tz_11)                 as tz_11
                                ,sum(T1.tz_12)                 as tz_12
                                ,sum(T1.tz_13)                 as tz_13
                                ,sum(T1.tz_14)                 as tz_14
                                ,sum(T1.tz_15)                 as tz_15
                                ,sum(T1.tz_16)                 as tz_16
                                ,sum(T1.tz_17)                 as tz_17
                                ,sum(T1.tz_18)                 as tz_18
                                ,sum(T1.tz_19)                 as tz_19
                                ,sum(T1.tz_20)                 as tz_20
                                ,sum(T1.tz_21)                 as tz_21
                                ,sum(T1.tz_22)                 as tz_22
                                ,sum(T1.tz_23)                 as tz_23
                            from {tbl_hrly} T1
                            left join (
                                {sql_cll_mppng_500}
                            ) T2 on T1.cell_id = T2.cell_50_id
                            group by T1.etl_ym 
                                   , T2.cell_500_id
                            order by T1.etl_ym
                                    ,T2.cell_500_id
                        '''
                    df_base   = pd.read_sql(sql_hrly_base, con = conn_postgresql)
                else:
                    if rsltn == '50':
                        sql_dly_base  = f'''
                            select
                                etl_ym
                                ,cell_id         
                                ,x_axis          
                                ,y_axis             
                                ,monday     
                                ,tuesday       
                                ,wednesday  
                                ,thursday   
                                ,friday    
                                ,saturday   
                                ,sunday    
                            from {tbl_dly} T1
                            where     etl_ym = '{YYYYMM}'
                                and cell_id in (
                                    {sql_cll_mppng_rgn}
                                )
                            order by etl_ym
                                    ,cell_id
                        '''
                    if rsltn == '500':
                        sql_dly_base  = f'''
                            select
                                T1.etl_ym                        as etl_ym
                                ,T2.cell_500_id                   as cell_id
                                ,avg(T1.x_axis)::numeric(6)       as x_axis   
                                ,avg(T1.y_axis)::numeric(6)       as y_axis      
                                ,sum(T1.monday)                   as monday    
                                ,sum(T1.tuesday)                  as tuesday  
                                ,sum(T1.wednesday)                as wednesday
                                ,sum(T1.thursday)                 as thursday
                                ,sum(T1.friday)                   as friday
                                ,sum(T1.saturday)                 as saturday
                                ,sum(T1.sunday)                   as sunday
                            from {tbl_dly} T1
                            left join (
                                {sql_cll_mppng_500}
                            ) T2 on T1.cell_id = T2.cell_50_id
                            group by T1.etl_ym 
                                    ,T2.cell_500_id
                            order by T1.etl_ym
                                    ,T2.cell_500_id
                        '''
                    
                    df_base   = pd.read_sql(sql_dly_base, con = conn_postgresql)
                    
                df_base   = df_base.astype(convert_dict_1).fillna(0)
                with open(f'{fol_pkl}/{tbl_name}_{rsltn}_{YYYYMM}.pkl','wb') as f:
                    pickle.dump(df_base,f)

                del df_base
                gc.collect()
                

                dend_tm = datetime.now(KST)
                del_tm  = dend_tm-dst_tm
                print(f'''[LOG] {tbl_name}_{rsltn}_{YYYYMM} 쿼리 마침, 마침시간 = {datetime.strftime(dend_tm, '%Y%m%d %H:%M:%S')}, 소요시간 = {del_tm}''') 

            dst_tm1      = datetime.now(KST)
            print(f'''[LOG] df_{tbl_name}_{rsltn}_{YYYYMM} 전처리 시작, 시작시간 = {datetime.strftime(dst_tm1, '%Y%m%d %H:%M:%S')}''') 

            ## 메모리 최적화
            dst_tm      = datetime.now(KST)
            print(f'''[LOG] df_{tbl_name}_{rsltn}_{YYYYMM} 메모리 최적화 및 Null값처리 시작, 시작시간 = {datetime.strftime(dst_tm, '%Y%m%d %H:%M:%S')}''') 

            # using dictionary to convert specific columns
            df_base_opt = pd.DataFrame()

            df_base_opt     = pd.read_pickle(f'{fol_pkl}/{tbl_name}_{rsltn}_{YYYYMM}.pkl')
            df_base_opt     = df_base_opt.rename(columns = convert_dict_1_a)

            dend_tm = datetime.now(KST)
            del_tm  = dend_tm-dst_tm
            print(f'''[LOG] df_{tbl_name}_{rsltn}_{YYYYMM} 메모리 최적화 및 Null값처리 마침, 마침시간 = {datetime.strftime(dend_tm, '%Y%m%d %H:%M:%S')}, 소요시간 = {del_tm}''')

            ## 데이터 Unpivot(Melt)
            dst_tm      = datetime.now(KST)
            print(f'''[LOG] Unpivot df_{tbl_name}_{rsltn}_{YYYYMM} 생성 시작, 시작시간 = {datetime.strftime(dst_tm, '%Y%m%d %H:%M:%S')}''') 

            if sq == 'hourly':
                df_base_opt_mlt = pd.melt(df_base_opt, id_vars=['etl_ym','cell_id','x_axis','y_axis'],var_name=f'hours',value_name='hourly_count')
            else:
                df_base_opt_mlt = pd.melt(df_base_opt, id_vars=['etl_ym','cell_id','x_axis','y_axis'],var_name=f'dayofweek',value_name='daily_count')

            del df_base_opt
            gc.collect()


            df_base_opt_mlt_opt = df_base_opt_mlt.astype(convert_dict_2).fillna(0)

            del df_base_opt_mlt
            gc.collect()

            dend_tm = datetime.now(KST)
            del_tm  = dend_tm-dst_tm
            print(f'''[LOG] Unpivot df_{tbl_name}_{rsltn}_{YYYYMM} 생성 마침, 마침시간 = {datetime.strftime(dend_tm, '%Y%m%d %H:%M:%S')}, 소요시간 = {del_tm}''')



            dst_tm        = datetime.now(KST)
            print(f'''[LOG] 한글컬럼 df_{tbl_name}_{rsltn}_{YYYYMM} 생성 시작, 시작시간 = {datetime.strftime(dst_tm, '%Y%m%d %H:%M:%S')}''')

            ## 컬럼명 한글컬럼 Rename
            df_base_opt_mlt_opt_kor = (
                df_base_opt_mlt_opt
                    .rename(columns = kor_dict
                    )
            )
            del df_base_opt_mlt_opt
            gc.collect()

            dend_tm = datetime.now(KST)
            del_tm  = dend_tm-dst_tm
            print(f'''[LOG] 한글컬럼 df_{tbl_name}_{rsltn}_{YYYYMM} 생성 마침, 마침시간 = {datetime.strftime(dend_tm, '%Y%m%d %H:%M:%S')}, 소요시간 = {del_tm}''')


            dst_tm      = datetime.now(KST)
            print(f'''[LOG] 한글컬럼 df_{tbl_name}_{rsltn}_{YYYYMM} Pickle 저장 및 메모리 해제 시작, 시작시간 = {datetime.strftime(dst_tm, '%Y%m%d %H:%M:%S')}''') 

            with open(f'{fol_pkl}/{tbl_name}_{rsltn}_kor_opt_{YYYYMM}.pkl','wb') as f:
                pickle.dump(df_base_opt_mlt_opt_kor,f)

            del df_base_opt_mlt_opt_kor
            gc.collect()

            dend_tm = datetime.now(KST)
            del_tm  = dend_tm-dst_tm
            print(f'''[LOG] 한글컬럼 df_{tbl_name}_{rsltn}_{YYYYMM} Pickle 저장 및 메모리 해제 마침, 마침시간 = {datetime.strftime(dend_tm, '%Y%m%d %H:%M:%S')}, 소요시간 = {del_tm}''')

            dend_tm1 = datetime.now(KST)
            del_tm1  = dend_tm1-dst_tm1
            print(f'''[LOG] df_{tbl_name}_{rsltn}_{YYYYMM} 전처리 마침, 마침시간 = {datetime.strftime(dend_tm1, '%Y%m%d %H:%M:%S')}, 전처리 총 소요시간 = {del_tm1}''')

    ################################################################################################################################################################ 
    ## 데이터 Time Value 형성
    ################################################################################################################################################################
    dst_tm      = datetime.now(KST)
    print(f'''[LOG] 데이터 Time Value 형성 시작, 시작시간 = {datetime.strftime(dst_tm, '%Y%m%d %H:%M:%S')}''') 

    df_timerange                = pd.date_range(strt_YYYYMMDD, end_YYYYMMDD, freq='D')
    df_times                    = pd.concat([pd.DataFrame(df_timerange.strftime("%Y%m").tolist()),pd.DataFrame(df_timerange.strftime('%A').tolist()),pd.DataFrame(df_timerange.days_in_month)], axis=1)
    df_times.columns            = ['기준년월','요일별','월일수']
    df_times['요일수']          = df_times.groupby(['기준년월','요일별'])['요일별'].transform('count')
    df_times                    = df_times.drop_duplicates()
    df_times_daily              = df_times[['기준년월','요일별','요일수']].drop_duplicates()
    df_times_daily['요일수']    = df_times_daily['요일수'].astype(np.uint8)
    df_times_daily['기준년월']  = df_times_daily['기준년월'].astype(np.uint32)
    # df_times_hourly             = df_times[['기준년월','월일수']].drop_duplicates()
    # df_times_hourly['기준년월'] = df_times_hourly['기준년월'].astype('int64')

    dend_tm = datetime.now(KST)
    del_tm  = dend_tm-dst_tm
    print(f'''[LOG] 데이터 Time Value 형성 마침, 마침시간 = {datetime.strftime(dend_tm, '%Y%m%d %H:%M:%S')}, 소요시간 = {del_tm}''')

    ####################################################################################################
    ###### 유동인구 Input Mart 형성 - 50X50, 500 X 500
    ####################################################################################################
    col_cll_id    = f'셀코드'
    col_cll_id_nm = f'격자인덱스'

    for idx2, YYYYMM in enumerate(list_YYYYMM):
        dst_tm      = datetime.now(KST)
        print(f'''[LOG] 요일별, 시간대별 마트 Import 시작, 시작시간 = {datetime.strftime(dst_tm, '%Y%m%d %H:%M:%S')}''') 
        df_dly_kor_opt                        = pd.read_pickle(f'{fol_pkl}/{tbl_dly}_{rsltn}_kor_opt_{YYYYMM}.pkl')
        df_hrly_kor_opt                       = pd.read_pickle(f'{fol_pkl}/{tbl_hrly}_{rsltn}_kor_opt_{YYYYMM}.pkl')
        df_hrly_kor_opt                       = df_hrly_kor_opt.astype({'기준년월' : np.uint32})

        dend_tm = datetime.now(KST)
        del_tm  = dend_tm-dst_tm
        print(f'''[LOG] 요일별, 시간대별 마트 Import 마침, 마침시간 = {datetime.strftime(dend_tm, '%Y%m%d %H:%M:%S')}, 소요시간 = {del_tm}''')

        dst_tm      = datetime.now(KST)
        print(f'''[LOG] 요일_시간대 통합 마트 형성 시작, 시작시간 = {datetime.strftime(dst_tm, '%Y%m%d %H:%M:%S')}''') 

        df_dly_kor_opt                        = df_dly_kor_opt.merge(df_times_daily,how = 'left')

        

        df_dly_kor_opt['요일별평균']          = df_dly_kor_opt[f'요일별집계값']/(df_dly_kor_opt['요일수'])
        df_hrly_kor_opt['유동인구월별집계']   = df_hrly_kor_opt.groupby(['기준년월',f'{col_cll_id}'])['시간대집계값'].transform('sum')
        df_hrly_kor_opt['월별시간대별가중치'] = df_hrly_kor_opt['시간대집계값']/(df_hrly_kor_opt['유동인구월별집계'])

        df_dly_hrly_base                      = df_dly_kor_opt.merge(df_hrly_kor_opt,how = 'left')
        
        del df_dly_kor_opt, df_hrly_kor_opt
        gc.collect()

        df_dly_hrly_base[f'유동인구수']            = df_dly_hrly_base['요일별평균']*df_dly_hrly_base['월별시간대별가중치']
        df_dly_hrly_base['요일']                   = df_dly_hrly_base['요일별'].replace({'Sunday' : 0, 'Monday' : 1,'Tuesday' : 2,'Wednesday' : 3,'Thursday' : 4,'Friday' : 5,'Saturday' : 6})
        df_dly_hrly_base_mart                      = df_dly_hrly_base[['기준년월','요일','시간대',f'{col_cll_id}',f'유동인구수']].sort_values(by=['기준년월','요일','시간대',f'{col_cll_id}']).drop_duplicates()

        with open(f'{fol_pkl}/mrt_inlay_flt_pop_input_{rsltn}_base_{YYYYMM}.pkl','wb') as f:
            pickle.dump(df_dly_hrly_base_mart,f)    

        del df_dly_hrly_base
        gc.collect()

        dend_tm = datetime.now(KST)
        del_tm  = dend_tm-dst_tm
        print(f'''[LOG] 요일_시간대 통합 마트 형성 마침, 마침시간 = {datetime.strftime(dend_tm, '%Y%m%d %H:%M:%S')}, 소요시간 = {del_tm}''')

        dst_tm      = datetime.now(KST)
        print(f'''[LOG] 셀코드 격자인덱스 매핑 마트 형성 시작, 시작시간 = {datetime.strftime(dst_tm, '%Y%m%d %H:%M:%S')}''') 

        
        df_dly_hrly_base_mart['기준시간대']       = df_dly_hrly_base_mart['기준년월'].astype(str)+ df_dly_hrly_base_mart['요일'].astype(str) + df_dly_hrly_base_mart['시간대'].astype(str).str.zfill(2)   
        df_dly_hrly_base_mart['시간대인덱스']     = df_dly_hrly_base_mart['기준시간대'].factorize()[0]
        df_dly_hrly_base_mart[f'{col_cll_id_nm}'] = df_dly_hrly_base_mart[f'{col_cll_id}'].factorize()[0]

        

        dend_tm = datetime.now(KST)
        del_tm  = dend_tm-dst_tm
        print(f'''[LOG] 셀코드 격자인덱스 매핑 마트 형성 마침, 마침시간 = {datetime.strftime(dend_tm, '%Y%m%d %H:%M:%S')}, 소요시간 = {del_tm}''')

        dst_tm      = datetime.now(KST)
        print(f'''[LOG] mrt_inlay_flt_pop_input_{rsltn}_{YYYYMM} 마트 형성 시작, 시작시간 = {datetime.strftime(dst_tm, '%Y%m%d %H:%M:%S')}''') 

        df_dly_hrly_base_mart                 = df_dly_hrly_base_mart[['기준년월',f'{col_cll_id}','시간대인덱스','시간대',f'{col_cll_id_nm}','요일',f'유동인구수']].sort_values(by=['기준년월','시간대인덱스',f'{col_cll_id_nm}']).drop_duplicates()

        with open(f'{fol_pkl}/mrt_inlay_flt_pop_input_{rsltn}_{YYYYMM}.pkl','wb') as f:
            pickle.dump(df_dly_hrly_base_mart,f)

        del df_dly_hrly_base_mart
        gc.collect()

        dend_tm = datetime.now(KST)
        del_tm  = dend_tm-dst_tm
        print(f'''[LOG] mrt_inlay_flt_pop_input_{rsltn}_{YYYYMM} 마트 형성 마침, 마침시간 = {datetime.strftime(dend_tm, '%Y%m%d %H:%M:%S')}, 소요시간 = {del_tm}''')

    del df_times_daily
    gc.collect()


    ####################################################################################################
    ###### 유동인구 데이터 적재 - Pandas(데이터프레임) --> SQL
    ####################################################################################################
    if inptmrt_db_upld:
        db_name     = postgresql_DB
        
        tbl_nm_inpt = flt_pop_ + rsltn

        dcrr_tm  = datetime.now(KST)
        etl_dt   = datetime.strftime(dcrr_tm, '%Y%m%d%H%M%S%f')[:-3]
        if rsltn == '50':
            char = 'id'
        else:
            char = '500'

        for idx1, YYYYMM in enumerate(list_YYYYMM):
            dst_tm  = datetime.now(KST)
            print(f'''[LOG] {db_name} DB {tbl_nm_inpt}_{YYYYMM} 데이터 삭제 시작, 시작시간 = {datetime.strftime(dst_tm, '%Y%m%d %H:%M:%S')}''')

            ## 데이터 초기화
            
            conn_postgresql = connect_postgresql(host, database, username, password, port)
            cur_postgresql  = conn_postgresql.cursor()
            cur_postgresql.execute(f"""delete from {tbl_nm_inpt} where baseym = '{YYYYMM}'""")

            dend_tm = datetime.now(KST)
            del_tm  = dend_tm-dst_tm
            print(f'''[LOG] {db_name} DB {tbl_nm_inpt}_{YYYYMM} 데이터 삭제 마침, 마침시간 = {datetime.strftime(dend_tm, '%Y%m%d %H:%M:%S')}, 소요시간 = {del_tm}''')

            ## 데이터 로드
            dst_tm  = datetime.now(KST)
            print(f'''[LOG] {tbl_nm_inpt}_{YYYYMM} 마트 Import 시작, 시작시간 = {datetime.strftime(dst_tm, '%Y%m%d %H:%M:%S')}''')

            df_dly_hrly_base_mart      = pd.read_pickle(f'{fol_pkl}/{tbl_nm_inpt}_{YYYYMM}.pkl')

            df_dly_hrly_base_mart      = df_dly_hrly_base_mart.loc[~df_dly_hrly_base_mart['셀코드'].isna()]

            dend_tm = datetime.now(KST)
            del_tm  = dend_tm-dst_tm
            print(f'''[LOG] {tbl_nm_inpt}_{YYYYMM} 마트 Import 마침, 마침시간 = {datetime.strftime(dend_tm, '%Y%m%d %H:%M:%S')}, 소요시간 = {del_tm}''')

            ## 데이터 적재
            list_tmp_pk = ['baseym', f'cell_500','tm_idx']

            dst_tm        = datetime.now(KST)
            print(f'''[LOG] {db_name} DB {tbl_nm_inpt}_{YYYYMM} 데이터 적재 시작, 시작시간 = {datetime.strftime(dst_tm, '%Y%m%d %H:%M:%S')}''')

            
            df_dly_hrly_base_mart.columns             = ['baseym',f'cell_{char}','tm_idx','timezn','cell_idx','weekday','flt_pop_500']
            df_dly_hrly_base_mart[f'flt_pop_500']      = np.round(df_dly_hrly_base_mart['flt_pop_500'],1)
            df_dly_hrly_base_mart['ETL_DT']           = etl_dt
            df_dly_hrly_base_mart           = df_dly_hrly_base_mart.astype(
                                                                            {
                                                                                 'baseym'       : object 
                                                                                ,f'cell_{char}' : object
                                                                                ,'tm_idx'       : np.int64
                                                                                ,'timezn'       : object
                                                                                ,'cell_idx'     : np.int64 
                                                                                ,'weekday'      : object
                                                                                ,'flt_pop_500'  : np.float64
                                                                                ,'ETL_DT'       : object
                                                                            }
                                                )
            
            df_dly_hrly_base_mart.to_sql(
                  name      = f'{tbl_nm_inpt}'
                , con       = engine
                , if_exists = 'append'
                , index     = False
                , method    = "multi"
                , chunksize = 10000
                )

            dend_tm = datetime.now(KST)
            del_tm  = dend_tm-dst_tm
            print(f'''[LOG] {db_name} DB {tbl_nm_inpt}_{YYYYMM} 데이터 적재 마침, 마침시간 = {datetime.strftime(dend_tm, '%Y%m%d %H:%M:%S')}, 소요시간 = {del_tm}''')

            ##적재 건수 확인

            DF건수 = len(df_dly_hrly_base_mart.drop_duplicates(list_tmp_pk))

            DB건수 = pd.read_sql(
                f"""
                    select count(1) as 건수
                    from {tbl_nm_inpt}
                    where baseym = '{YYYYMM}'
                """
            , conn_postgresql
            ).values[0][0]

            print(rf"[LOG] DF건수 = {DF건수:,}, DB건수 = {DB건수:,}")

            del df_dly_hrly_base_mart
            gc.collect()