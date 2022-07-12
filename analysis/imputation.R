setwd("~/nfs_fs02/imputation/")
library(tidyverse)
library(missForest)
library(doParallel)
library(lubridate)

pre<-readRDS("temps")%>%drop_na()%>%spread(site,temp)%>%select(-nerr)%>%mutate(missing=rowSums(is.na(.))/30)%>%filter(missing<0.3)%>%select(-missing)%>%column_to_rownames(var="date_time")
formatted<-t(as.matrix(pre))
registerDoParallel(cores=16)
imputed<-missForest(formatted,mtry=100,parallelize = 'variables')

saveRDS(imputed, "imputed");print('done saving imputed')

check<-pre%>%rownames_to_column(var="date_time")%>%mutate(date_time=ymd_hms(date_time))%>%gather(site,temp,-date_time)
finalized<-as.data.frame(t(imputed$ximp))%>%rownames_to_column(var="date_time")%>%mutate(date_time=ymd_hms(date_time))%>%
     gather(site,temp,-date_time)%>%left_join(.,check,by=c("site","date_time"))%>%
     mutate(test = replace_na(temp.y,'imputed'))%>%mutate(test=case_when(test!='imputed'~'original',TRUE~as.character(test)))%>%
     select(-temp.y)%>%rename(temp=temp.x)

saveRDS(finalized,"final_imputed");print('done saving final')
