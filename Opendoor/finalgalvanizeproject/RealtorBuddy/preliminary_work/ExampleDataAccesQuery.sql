with no_duplicate_homes as --- select only the last listing of each home 
( 
           select     * 
           from       listings 
           right join 
                      ( 
                                 select     max(listings.id) as id2 
                                 from       listings 
                                 right join 
                                            ( 
                                                     select   listid            as listid, 
                                                              originallistprice as originallistprice,
                                                              max(modtimestamp) as modtimestamp 
                                                     from     listings 
                                                     group by listid, 
                                                              originallistprice, 
                                                              modtimestamp) as maxtime 
                                 on         listings.listid = maxtime.listid 
                                 and        listings.originallistprice = maxtime.originallistprice
                                 and        listings.modtimestamp = maxtime.modtimestamp 
                                 group by   listings.listid, 
                                            listings.originallistprice, 
                                            listings.modtimestamp) as maxtimeid 
           on         listings.id = maxtimeid.id2), market_homes as --- filter by criteria that defines a market home 
( 
       select *, 
              no_duplicate_homes.listprice::float - no_duplicate_homes.closeprice::float as pricedifference 
              coalesce(array_length(regexp_split_to_array(no_duplicate_homes.otherrooms, ','), 1),0) as numotherrooms,
              case 
                     when no_duplicate_homes.hoapaidfreq='monthly' then no_duplicate_homes.hoafee::float       *12
                     when no_duplicate_homes.hoapaidfreq='semi-annually' then no_duplicate_homes.hoafee::float *2 
                     when no_duplicate_homes.hoapaidfreq='quarterly' then no_duplicate_homes.hoafee::float     *4
                     when no_duplicate_homes.hoapaidfreq='annually' then no_duplicate_homes.hoafee::float      *1
                     else 0 
              end as hoafee1total, 
              case 
                     when no_duplicate_homes.hoa2paidfreq='monthly' then no_duplicate_homes.hoa2fee::float       *12
                     when no_duplicate_homes.hoa2paidfreq='semi-annually' then no_duplicate_homes.hoa2fee::float *2 
                     when no_duplicate_homes.hoa2paidfreq='quarterly' then no_duplicate_homes.hoa2fee::float     *4
                     when no_duplicate_homes.hoa2paidfreq='annually' then no_duplicate_homes.hoa2fee::float      *1
                     else 0 
              end                                                                                    as hoafee2total,
              extract(year from no_duplicate_homes.listdate::date)-no_duplicate_homes.yearbuilt      as age, 
              no_duplicate_homes.contractdate::date               -no_duplicate_homes.listdate::date as daytocontract 
       from   no_duplicate_homes 
       where  (( 
                            no_dupl icate_homes.county='xx' 
                     or     no_duplicate_homes.county='xx') 
              and    no_duplicate_homes.geolat >= xx 
              and    no_duplicate_homes.geolat <= xx 
              and    no_duplicate_homes.geolon >= xx 
              and    no_duplicate_homes.geolon <= xx 
              and    no_duplicate_homes.statuschangedate::date > (date '2009-01-01') 
              and    no_duplicate_homes.exteriorfeatures not ilike all (array['%racquetball%','%pvt tennis court%','%separate guest house%'])
              and    not (( 
                                   no_duplicate_homes.semiprivateremarks ilike '%short sale%' 
                            and    no_duplicate_homes.semiprivateremarks not ilike all (array['%or short%','%short sale or%','%not a short sale%','%not short sale%','%not a teaser%','%not teaser%','%not a teaser%','%not bank%','%not a bank%','%not flip%','%not a flip%','%not foreclosure%','%not a foreclosure%'])))
              and    not (( 
                                   no_duplicate_homes.semiprivateremarks ilike '%hud sale%' 
                            and    no_duplicate_homes.semiprivateremarks not ilike all (array['%or short%','%short sale or%','%not a short sale%','%not short sale%','%not a teaser%','%not teaser%','%not a teaser%','%not bank%','%not a bank%','%not flip%','%not a flip%','%not foreclosure%','%not a foreclosure%'])))
              and    not (( 
                                   no_duplicate_homes.remarks ilike '%short sale%' 
                            and    no_duplicate_homes.semiprivateremarks not ilike all (array['%or short%','%short sale or%','%not a short sale%','%not short sale%','%not a teaser%','%not teaser%','%not a teaser%','%not bank%','%not a bank%','%not flip%','%not a flip%','%not foreclosure%','%not a foreclosure%'])))
              and    not (( 
                                   no_duplicate_homes.remarks ilike '%hud sale%' 
                            and    no_duplicate_homes.semiprivateremarks not ilike all (array['%or short%','%short sale or%','%not a short sale%','%not short sale%','%not a teaser%','%not teaser%','%not a teaser%','%not bank%','%not a bank%','%not flip%','%not a flip%','%not foreclosure%','%not a foreclosure%'])))
              and    no_duplicate_homes.lotsize not ilike '%acres%' 
              and    no_duplicate_homes.auction != 'yes' 
              and    ( 
                            no_duplicate_homes.guesthousesqft is null 
                     or     no_duplicate_homes.guesthousesqft::float > 1.)) 
       and    coalesce(no_duplicate_homes.exteriorstories::float,0) < 5 
       and    coalesce(no_duplicate_homes.numinteriorlevels::float,0) < 5 
       and    coalesce(no_duplicate_homes.hoafee::float,0) < 5000 
       and    coalesce(no_duplicate_homes.hoa2fee::float,0) < 5000 
       and    coalesce(no_duplicate_homes.numbaths::float,0) < 10 
       and    coalesce(no_duplicate_homes.numbedrooms::float,0) < 10 
       and    coalesce(no_duplicate_homes.garagespaces::float,0) < 6 
       and    coalesce(no_duplicate_homes.slabparkingspaces::float,0) < 6 
       and    coalesce(no_duplicate_homes.totalcoveredspaces::float,0) < 6 
       and    coalesce(no_duplicate_homes.livingarea::float,0) > 500 
       and    coalesce(no_duplicate_homes.approxlotsqft::float,0) > 500 
       and    coalesce(no_duplicate_homes.numbedrooms::float,0) > 0 
       and    coalesce(no_duplicate_homes.numbaths::float,0) > 0 
       and    coalesce(no_duplicate_homes.taxes::float,0) > 0 
       and    no_duplicate_homes.listdate < no_duplicate_homes.closedate 
       and    no_duplicate_homes.listdate < no_duplicate_homes.contractdate), block_group_statistics as --- compute statistics for each tract 
( 
         select   market_homes2.countytractblock_group, 
                  stddev_pop(market_homes2.closeprice)    std_closeprice_block_group, 
                  median(market_homes2.closeprice)        median_closeprice_block_group,                  
                  avg(market_homes2.closeprice)           mean_closeprice_block_group, 
                  min(market_homes2.closeprice)           max_closeprice_block_group, 
                  max(market_homes2.closeprice)           min_closeprice_block_group,
                  stddev_pop(market_homes2.contract)    std_contract_block_group, 
                  median(market_homes2.contract)        median_contract_block_group, 
                  avg(market_homes2.contract)           average_contract_block_group, 
                  max(market_homes2.contract)           max_contract_block_group, 
                  min(market_homes2.contract)           min_contract_block_group,
                  market_homes2.listdate               as listdate 
         from     market_homes                         as market_homes1 
         join     market_homes                         as market_homes2 
         on       market_homes2.listdate::date < market_homes1.statuschangedate::date 
         and      market_homes2.listdate::date > (date '2009-01-01') 
         and      market_homes1.countytractblock_group = market_homes2.countytractblock_group 
         group by market_homes2.countytractblock_group, 
                  market_homes1.listdate), tract_statistics_full as --- compute statistics for each block group 
( 
         select   market_homes2.countytract, 
                  stddev_pop(market_homes2.closeprice)    std_closeprice_tract_f, 
                  median(market_homes2.closeprice)        median_closeprice_tract_f,                  
                  avg(market_homes2.closeprice)           mean_closeprice_tract_f, 
                  min(market_homes2.closeprice)           max_closeprice_tract_f, 
                  max(market_homes2.closeprice)           min_closeprice_tract_f, 
                  median(market_homes2.contract)        median_contract_tract_f, 
                  avg(market_homes2.contract)           average_contract_tract_f, 
                  max(market_homes2.contract)           max_contract_tract_f, 
                  min(market_homes2.contract)           min_contract_tract_f, 
                  market_homes2.listdate               as listdate 
         from     market_homes                         as market_homes1 
         join     market_homes                         as market_homes2 
         on       market_homes2.listdate::date < market_homes1.listdate::date 
         and      market_homes2.listdate::date > (date '2009-01-01')  
         and      market_homes1.countytract = market_homes2.countytract 
         group by market_homes2.countytract, 
                  market_homes1.listdate), block_group_statistics_full as --- compute statistics for each tract 
( 
         select   market_homes2.countytractblock_group, 
                  stddev_pop(market_homes2.closeprice)    std_closeprice_block_group_f, 
                  median(market_homes2.closeprice)        median_closeprice_block_group_f,                  
                  avg(market_homes2.closeprice)           mean_closeprice_block_group_f, 
                  min(market_homes2.closeprice)           max_closeprice_block_group_f, 
                  max(market_homes2.closeprice)           min_closeprice_block_group_f,
                  stddev_pop(market_homes2.contract)    std_contract_block_group_f, 
                  median(market_homes2.contract)        median_contract_block_group_f, 
                  avg(market_homes2.contract)           average_contract_block_group_f, 
                  max(market_homes2.contract)           max_contract_block_group_f, 
                  min(market_homes2.contract)           min_contract_block_group_f,
                  market_homes2.listdate               as listdate 
         from     market_homes                         as market_homes1 
         join     market_homes                         as market_homes2 
         on       market_homes2.listdate::date < market_homes1.statuschangedate::date 
         and      market_homes2.listdate::date > (market_homes1.statuschangedate)::date-xxxxx --- change for look back 
         and      market_homes1.countytractblock_group = market_homes2.countytractblock_group 
         group by market_homes2.countytractblock_group, 
                  market_homes1.listdate), tract_statistics as --- compute statistics for each block group 
( 
         select   market_homes2.countytract, 
                  stddev_pop(market_homes2.closeprice)    std_closeprice_tract, 
                  median(market_homes2.closeprice)        median_closeprice_tract,                  
                  avg(market_homes2.closeprice)           mean_closeprice_tract, 
                  min(market_homes2.closeprice)           max_closeprice_tract, 
                  max(market_homes2.closeprice)           min_closeprice_tract, 
                  median(market_homes2.contract)        median_contract_tract, 
                  avg(market_homes2.contract)           average_contract_tract, 
                  max(market_homes2.contract)           max_contract_tract, 
                  min(market_homes2.contract)           min_contract_tract, 
                  market_homes2.listdate               as listdate 
         from     market_homes                         as market_homes1 
         join     market_homes                         as market_homes2 
         on       market_homes2.listdate::date < market_homes1.listdate::date 
         and      market_homes2.listdate::date > (market_homes1.listdate)::date-xxxxx --- change for look back 
         and      market_homes1.countytract = market_homes2.countytract 
         group by market_homes2.countytract, 
                  market_homes1.listdate) final_table as --- create new features from columns 
( 
          select    * 
          from      market_homes 
          left join block_group_statistics 
          on        block_group_statistics.countytractblock_group=market_homes.countytractblock_group
          and       block_group_statistics.listdate=market_homes.listdate 
          left join tract_statistics 
          on        tract_statistics.countytract=market_homes.countytract 
          and       tract_statistics.listdate=market_homes.listdate 
          left join block_group_statistics_full 
          on        block_group_statistics.countytractblock_group=market_homes.countytractblock_group
          and       block_group_statistics.listdate=market_homes.listdate 
          left join tract_statistics_full 
          on        tract_statistics.countytract=market_homes.countytract 
          and       tract_statistics.listdate=market_homes.listdate
          where     no_duplicate_homes.contractdate::date-no_duplicate_homes.listdate::date>=0
          and       no_duplicate_homes.statuschangedate::date-no_duplicate_homes.listdate::date>=0)