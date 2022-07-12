setwd("~/omics/mcap_clonality/environmentaldata//")
library(tidyverse)
library(missForest)
library(doParallel)
library(lubridate)
library(plotrix)


set.seed(3839)
pre<-readRDS("temps")%>%drop_na()%>%spread(site,temp)%>%select(-nerr)%>%mutate(missing=rowSums(is.na(.))/30)%>%filter(missing<0.3)%>%select(-missing)%>%column_to_rownames(var="date_time")
working<-pre%>%sample_n(1000)
nonmissing<-working%>%rownames_to_column(var="datetime")%>%gather(site,temp,-datetime)
missing<-working%>%rownames_to_column(var="datetime")%>%gather(site,temp,-datetime)%>%
  rowwise()%>%
  mutate(rand=sample(0:100,1))%>%
  mutate(temp=case_when(rand>=95~NA_real_,
                    TRUE~as.numeric(temp)))

forimputation<-missing%>%select(-rand)%>%
  ungroup()%>%
  spread(site,temp)%>%
  column_to_rownames(var="datetime")

formatted<-t(as.matrix(forimputation))
registerDoParallel(cores=16)
set.seed(3839);imputed<-missForest(formatted,mtry=100,parallelize = 'variables')

output<-as.data.frame(t(imputed$ximp))%>%rownames_to_column(var="datetime")%>%gather(site,temp,-datetime)%>%
  rename(imputed_temp=temp)%>%
  left_join(.,nonmissing,by=c("datetime","site"))%>%
  left_join(.,missing,by=c("datetime","site"))%>%
  mutate(diff=(temp.x-imputed_temp))%>%
  mutate(percent_diff=diff/temp.x)%>%
  filter(rand>=87)%>%
  mutate(thresh=case_when(diff<0.1~"good",TRUE~"bad"))%>%
  filter(diff<2)

3242/(431+3242)
table(output$thresh)
mean(abs(output$diff),na.rm=TRUE)
std.error(output$diff)

quartz()
ggplot(output)+geom_histogram(aes(diff),bins=50)+
  theme_classic()+
  scale_x_continuous(limits=c(-2,2),breaks=seq(-2,2,0.5))+
  ylab("Count")+xlab("True-Imputed Difference")+
  annotate("text",x=-2,y=2000,label="0.07 +/- 0.002°C\n(abs.mean +/- 1SD)",size=3,hjust=0)


