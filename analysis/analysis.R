library(tidyverse);library(janitor);library(cowplot);library(readxl);library(lubridate)
library(ade4)
library(broom)
library(geodist)
library(ggdendro)
library(R.matlab)
library(zoo)
library(scales)
library(tools)
library(vegan)
library(ggrepel)
library(pdftools)
library(magick)
library(forcats)
library(factoextra)
options(scipen=999)

###################### MAP ########################################################### #####
library(sf);library(ggrepel);library(ggsn);library(cowplot);library(tidyverse);library(patchwork);library(jpeg);library(ggnewscale)

oahu<-st_transform(st_read("./data/oahu_map/coast_n83.shp"),crs=4386)
cropped<-st_crop(oahu,xmin=-158.8,xmax=-157.4,ymin=20.8,ymax=21.8)
fringe<-st_read("./data/oahu_map/Fringing Reef.shp")%>%select(id,geometry)%>%mutate(zone='Fringing Reef')
habitat<-st_read("./data/oahu_map/haw_benthic_habitat.shp")%>%clean_names()%>%filter(zone=='Backreef'|zone=="Reef Flat")%>%select(id,zone,geometry)
patch<-st_read("./data/oahu_map/Patches2.shp")%>%mutate(type="Reef")%>%clean_names()%>%select(id,zone,block,geometry)%>%mutate(zone='Patch Reef')
points<-read_tsv("./data/metadata/Site_depth_lat_long_numColonies.txt")%>%clean_names()%>%select(site,block,lat,lon)%>%mutate(block=paste("Block",as.factor(block)))

out<-bind_rows(st_transform(fringe,crs=4386),st_transform(habitat,crs=4386))%>%
     mutate(zone=case_when(zone=="Reef Flat"~ "Fringing Reef",
                           zone=="Reef Crest"~"Fringing Reef",
                           zone=="Backreef"|zone=="Fringing Reef"~"Back/Fringing Reef",
                           TRUE~as.character(zone)))
inset<-ggplotGrob(ggplot()+
                       geom_sf(data=cropped)+
                       theme_minimal(base_size=8)+
                       theme(axis.text.x=element_blank(),
                             axis.title.y=element_blank(),
                             axis.title.x=element_blank(),
                             axis.text.y=element_blank(),
                             panel.grid.minor=element_blank(),
                             panel.grid.major=element_blank(),
                             panel.border = element_rect(colour = "gray35", fill=NA, size=1),
                             panel.background = element_rect(color="white"))+
                       annotate("rect",xmin=-157.85,xmax=-157.76,ymin=21.413,ymax=21.507,color="blue",alpha=0.25,size=0.25)+
                       annotate("text",x=-158.05,y=21.52,label="Oʻahu",size=2.5))

map<-ggplot()+
     geom_sf(data=out,fill="lightgrey",color='lightgrey')+
     geom_sf(data=oahu,fill="darkgrey")+
     geom_sf(aes(fill=block,color=block),dat=patch)+
     #geom_label_repel(aes(lon,lat,label=site),data=points,size=2,min.segment.length = 0,nudge_x=0,segment.color = "grey50")+
     coord_sf(xlim=c(-157.85,-157.76),ylim=c(21.413,21.507))+
     theme_classic(base_size=8)+
     theme(legend.position=c(0.16,0.22),
           legend.key.size=unit(0.4,"cm"),
           axis.title=element_blank(),
           axis.text.y=element_text(angle=90,hjust=0.5))+
     scale_fill_viridis_d(direction=1,name="Block")+
     scale_color_viridis_d(direction=1,name="Block",guide=FALSE)+
     scalebar(x.min=-157.78,x.max=-157.82,y.min=21.415,y.max=21.50,transform=TRUE,dist=1,dist_unit="km",
              height = .02, st.dist = 0.04,
              box.fill = c("black", "white"),
              box.color = "black", border.size = .1,st.size=2,
              location="bottomleft")+
     annotation_custom(inset,xmin=-157.8,xmax=-157.75,ymin=21.479,ymax=21.512)+
     #new_scale_color()+
     geom_point(aes(lon,lat,fill=block),pch=21,data=points)+
     scale_fill_viridis_d(name="Block");map

#site example
meta<-read_tsv("./data/coraldata/All_640_Samples_Site_Info_sortedBySample.txt")%>%clean_names()%>%mutate(sample=as.factor(sample))%>%filter(sample!='47B',sample!='479B')%>%mutate(sample=str_remove(sample,'B'))
summary_clonality<-read_tsv("./data/coraldata/summary_clonality.txt")%>%select(sample,CLUSTER)
use<-meta%>%filter(site=='1_3')%>%
     filter(!str_detect(sample,"a|b"))%>%
     left_join(.,summary_clonality%>%mutate(sample=as.character(sample)),by="sample")%>%
     clean_names()%>%
     group_by(cluster)%>%mutate(n=n())%>%
     mutate(cluster=case_when(n==2~"A",
                              n==8~"B",
                              TRUE~"Nonclonal"))%>%
     filter(abs(x)<6,abs(y)<6)

ggplot(use)+
     geom_point(aes(x,y),pch=21,size=3)+
     scale_y_continuous(limits=c(-10,10.5))+
     scale_x_continuous(limits=c(-10,10))+
     annotate("segment",x=-6,xend=0,y=0,yend=0)
     
img <- image_flop(image_read_pdf("./data/mosaics/site_2_01_time_04.pdf"))

a<-ggplot(use)+
     geom_hline(yintercept=0,color='white',size=15,alpha=0.4)+
     geom_vline(xintercept=0,color='white',size=15,alpha=0.4)+
     geom_hline(yintercept=0,color='white')+
     geom_vline(xintercept=0,color='white')+
     theme_classic(base_size=8)+
     scale_y_continuous(limits=c(-6,6))+
     scale_x_continuous(limits=c(-6,6))+
     theme(legend.position=c(0,0.1),
           legend.key.size=unit(0.2,"cm"),
           legend.background = element_rect(fill="white"),
           plot.background = element_blank(),
           panel.background=element_blank(),
           axis.ticks=element_blank(),
           axis.title=element_blank(),
           axis.text=element_blank(),
           axis.line=element_blank())+
     annotate("text",x=8.3,y=8.3,label="Site 5_6",fontface="italic",size=3)+
     annotate("text",x=-5,y=-0.5,label="6m",size=3,color='white',fontface='bold')+
     #annotate("segment",x=-6,xend=6,y=0,yend=0,linetype="dotted")+
     #annotate("segment",x=0,xend=0,y=-6,yend=6,linetype="dotted")+
     geom_point(aes(x,y),pch=21,size=3,fill='orange')

b<-ggdraw()+
     draw_image(img,scale=1,x=0.02,width=0.95,y=0,height=1)+
     draw_plot(a);b

quartz(w=6,h=2.9) 
plot_grid(map,b,align="v",axis="tb",labels=c("A","B"),label_size=8)

#supplemental
quartz(w=6,h=7)
sf1<-ggplot()+
     geom_sf(data=out,fill="lightgrey",color='lightgrey')+
     geom_sf(data=oahu,fill="darkgrey")+
     geom_sf(aes(fill=block,color=block),dat=patch)+
     geom_label_repel(aes(lon,lat,label=site),data=points,size=3,min.segment.length = 0,nudge_x=0,segment.color = "grey50",max.overlaps = Inf)+
     coord_sf(xlim=c(-157.85,-157.76),ylim=c(21.413,21.507))+
     theme_classic(base_size=10)+
     theme(legend.position=c(0.075,0.1),
           legend.key.size=unit(0.4,"cm"),
           axis.title=element_blank(),
           axis.text.y=element_text(angle=90,hjust=0.5))+
     scale_fill_viridis_d(direction=1,name="Block")+
     scale_color_viridis_d(direction=1,name="Block")+
     geom_point(aes(lon,lat),pch=21,data=points);sf1

###################### TEMPERATURE IMPUTATION ######################################## #####
# setwd("~/Projects/K_Bay_SurveySites_Temperature/readydata")
# library(tidyverse);library(janitor);library(lubridate);library(tools);library(missForest)
# library(zoo);library(scales)
# 
# #dataprep
# rm(list=ls(pattern="*"))
# list<-as.vector(list.files(pattern="*.csv"));list #create list of csv files
# for (i in list){
#      assign(paste0("data",file_path_sans_ext(i,compression=TRUE)),read_csv(i)%>%clean_names()%>%select(date_time,3)%>%mutate(date_time=mdy_hm(date_time)))
# }
# 
# data<-reduce(mget(ls(pattern = "data.+")), full_join, by ="date_time")
# output<-data%>%gather(site,temp,-date_time)%>%separate(site,into=c("site","trash"),sep="[.]")%>%select(-trash) #several minutes to run
# saveRDS(output,"~/CRD_GBS/mcap_clonality/environmentaldata/temps")
# 
# #imputation -- RUN ON HPC
# setwd("~/nfs_fs02/imputation/")
# library(tidyverse)
# library(missForest)
# library(doParallel)
# library(lubridate)
# 
# pre<-readRDS("temps")%>%drop_na()%>%spread(site,temp)%>%select(-nerr)%>%mutate(missing=rowSums(is.na(.))/30)%>%filter(missing<0.3)%>%select(-missing)%>%column_to_rownames(var="date_time")
# formatted<-t(as.matrix(pre))
# registerDoParallel(cores=16)
# imputed<-missForest(formatted,mtry=100,parallelize = 'variables')
# 
# saveRDS(imputed, "imputed")
# 
# check<-pre%>%rownames_to_column(var="date_time")%>%mutate(date_time=ymd_hms(date_time))%>%gather(site,temp,-date_time)
# finalized<-as.data.frame(t(imputed$ximp))%>%rownames_to_column(var="date_time")%>%mutate(date_time=ymd_hms(date_time))%>%
#      gather(site,temp,-date_time)%>%left_join(.,check,by=c("site","date_time"))%>%
#      mutate(test = replace_na(temp.y,'imputed'))%>%mutate(test=case_when(test!='imputed'~'original',TRUE~as.character(test)))%>%
#      select(-temp.y)%>%rename(temp=temp.x)
# 
# saveRDS(finalized,"final_imputed")

raw<-readRDS("./data/environmentaldata/final_imputed")%>%mutate(temp=case_when(temp<18~NA_real_,TRUE~as.numeric(temp)))%>%
     mutate(date=date(date_time),hour=hour(date_time),year=year(date_time))%>%filter(date!='2019-05-21',date!='2019-05-20')#removed two problematic days near end

#table(fulltemp$test) #305195/(305195+2068015) 12.86% of data imputed
percent<-raw%>%group_by(site,test)%>%add_tally()%>%select(site,test,n)%>%distinct()%>%spread(test,n)%>%
     mutate(imputed= replace_na(imputed, 0))%>%mutate(prop=imputed/(imputed+original))

full_summary<-raw%>%group_by(site,date,hour)%>%summarise(temp=mean(temp))%>% #first take hourly average
     mutate(accumulated=case_when(temp>28.5~temp-28.5,TRUE~0))%>%
     group_by(site,date)%>%mutate(range=max(temp)-min(temp))%>%
     group_by(date,hour)%>%mutate(bay_avg=mean(temp))%>%
     ungroup()%>%mutate(residual=temp-bay_avg)%>%
     group_by(site,hour)%>%mutate(hots=case_when(temp>=30~1,temp<30~0))%>%
     group_by(site)%>%summarise(overall_min=min(temp),
                                overall_max=max(temp),
                                overall_mean=mean(temp),
                                overall_sd=sd(temp),
                                overall_mean_daily_range=mean(range),
                                overall_residual=mean(residual),
                                hrs_above_30=sum(hots),
                                total_DHW=sum(accumulated)/24/7/3)

summer_summary<-raw%>%group_by(site,date,hour)%>%summarise(temp=mean(temp))%>%
     filter((date>='2017-08-15'&date<='2017-10-15')|(date>='2018-08-15'&date<='2018-10-15')|(date>='2019-08-15'&date<='2019-10-15'))%>%
     mutate(accumulated=case_when(temp>28.5~temp-28.5,TRUE~0))%>%
     group_by(site,date)%>%mutate(range=max(temp)-min(temp))%>%
     group_by(date,hour)%>%mutate(bay_avg=mean(temp))%>%
     ungroup()%>%mutate(residual=temp-bay_avg)%>%
     group_by(site)%>%summarise(summer_min=min(temp),
                                summer_mean=mean(temp),
                                summer_sd=sd(temp),
                                summer_mean_daily_range=mean(range))

temp_summary<-left_join(full_summary,summer_summary,by="site")%>%separate(site,into=c("trash","site"),sep=1)%>%select(-trash)

###################### ENVIRO DATA ################################################### #####
meta<-read_xlsx("./data/metadata/Coral_Sample_Metadata.xlsx")%>%clean_names()%>%select(site,id)
velo<-as.data.frame(t(readMat("./data/environmentaldata/AllKbaySameTimes_March2019.mat")$Vprime))%>%rename(KB1=1,KB2=2,KB3=3,KB4=4,KB5=5,KB6=6)%>%select(-KB6)%>%gather(site,rms)
height<-as.data.frame(t(readMat("./data/environmentaldata/AllKbaySameTimes_March2019.mat")$WaveHs))%>%rename(KB1=1,KB2=2,KB3=3,KB4=4,KB5=5,KB6=6)%>%select(-KB6)%>%gather(site,height)

waves_plotdata<-bind_cols(velo,height)%>%rename(site=1)%>%select(-3)%>%separate(site,into=c("garb","block"),sep=2)

sed_plotdata<-read_xlsx("./data/environmentaldata/mastersedimentation.xlsx")%>%clean_names()%>%
     select(block,site,contains('day'))%>%
     gather(rep,sed,-block,-site)%>%
     mutate(sed=sqrt(sed))%>%
     mutate(block=as.factor(block))

sed_summary<-read_xlsx("./data/environmentaldata/mastersedimentation.xlsx")%>%clean_names()%>%
     select(block,site,contains('day'))%>%
     gather(rep,sed,-block,-site)%>%
     mutate(sed=(sed))%>%
     mutate(block=as.factor(block))%>%drop_na()%>%group_by(site)%>%mutate(sed_mean=mean(sed),sed_max=max(sed),sed_sd=sd(sed))%>%select(block,site,sed_mean,sed_max,sed_sd)%>%distinct()

wave_summary<-bind_cols(velo,height)%>%rename(site=1)%>%select(-3)%>%separate(site,into=c("garb","block"),sep=2)%>%select(-garb)%>%group_by(block)%>%
     summarise(rms_mean=mean(rms),rms_sd=sd(rms),height_mean=mean(height),height_sd=sd(height))

depth_summary<-read_tsv("./data/environmentaldata/depth.txt")%>%drop_na()%>%clean_names()

summary<-full_join(full_join(depth_summary,sed_summary,by="site"),wave_summary,by="block")%>%
     right_join(.,meta,by='site')%>%mutate(sample=as.numeric(id))%>%
     select(sample,block,site,everything(),-id)%>%arrange(sample)%>%
     left_join(.,temp_summary,by="site")
#saveRDS(summary,"./data/environmentaldata/environmental_summary")

table1<-full_join(full_join(depth_summary,sed_summary,by="site"),wave_summary,by="block")%>%
     left_join(.,temp_summary,by="site")%>%
     select(site,block,everything(),-sed_sd,-rms_sd)%>%
     select(-sed_max,-height_sd,-overall_mean,-overall_sd)%>%
     clean_names(case="title")%>%
     mutate(across(4:11, round, 3))%>%
     mutate(across(12:16, round, 3))
write.table(table1,"./manuscript/table1.txt",sep="\t",quote=FALSE,row.names=FALSE)

###################### PLOTS ######################################################### #####
envirodata<-readRDS('./data/environmentaldata/environmental_summary')%>%select(-sample)%>%distinct()%>%arrange(site)
raw<-readRDS("./data/environmentaldata/final_imputed")%>%mutate(temp=case_when(temp<18~NA_real_,TRUE~as.numeric(temp)))%>%
     mutate(date=date(date_time),hour=hour(date_time),year=year(date_time))%>%filter(date!='2019-05-21',date!='2019-05-20')%>%#removed two problematic days near end
     select(date_time,site,temp)%>%spread(date_time,temp)%>%
     arrange(site)%>%
     column_to_rownames(var="site")

pca_data<-as.matrix(raw)
pca<-prcomp(pca_data)
summary(pca)
axes<-fviz_pca_ind(pca,axes = c(1,2))
pca_plotdata<-as.data.frame(axes$data)%>%rename(id=1)%>%bind_cols(.,envirodata)
temp_plotdata<-readRDS("./data/environmentaldata/final_imputed")%>%mutate(temp=case_when(temp<18~NA_real_,TRUE~as.numeric(temp)))%>%
     mutate(date=date(date_time),hour=hour(date_time),year=year(date_time))%>%filter(date!='2019-05-21',date!='2019-05-20')%>%#removed two problematic days near end due to bad data in 2_2 during May 2019
     mutate(period=case_when(date<=as.Date('2017-12-01')~'a1',
                             date>as.Date('2017-12-1')&date<as.Date('2018-06-01')~'a2',
                             date>=as.Date('2018-06-02')&date<as.Date('2019-03-31')~'a3',
                             date>as.Date('2019-04-01')~'a4'))%>%
     group_by(site,date,year)%>%mutate(meantemp=mean(temp))%>%
     separate(site,into=c('garb','site'),sep=1)%>%select(-garb)%>%
     separate(site,into=c("block","garb"),sep="_",remove=FALSE)%>%select(-garb)%>%
     mutate(drop=case_when((site=="2_2"&test=="original"&period=="a4")~'drop',TRUE~'keep'))%>%
     filter(drop!='drop')%>%select(-drop)
     
pca<-ggplot(pca_plotdata)+
     stat_ellipse(aes(x,y,color=block))+
     geom_point(aes(x,y,color=block),size=1)+
     theme_classic(base_size=8)+
     scale_color_viridis_d()+
     xlab("PC1 (35.2%)")+ylab("PC2 (14.8%)")+
     theme(legend.position="none")+
     annotate("text",x=200,y=125,label="Temperature\nData",fontface="italic",color="darkgrey",size=2,hjust=1)

timeseries<-ggplot(temp_plotdata)+
     geom_hline(yintercept=28.5,linetype="dotted",color='black')+
     geom_line(aes(date_time,meantemp,group=interaction(site,period),color=block),size=0.5,alpha=1)+
     theme_classic(base_size=8)+
     theme(legend.position=c(0.95,0.3),
           legend.key.size=unit(0.25,"cm"),
           legend.key = element_blank(),
           legend.background = element_blank(),
           axis.title.x=element_blank())+
     scale_x_datetime(date_breaks="2 months",minor_breaks=waiver(),labels=label_date_short())+
     scale_color_viridis_d(name="Block")+
     ylab("Mean Temperature\n(daily,°C)")+
     scale_y_continuous(limits=c(20,30),breaks=seq(20,30,1))+
     annotate("text",as_datetime('2017-08-15'),y=30,label="30 sites",fontface="italic",color="darkgrey",size=2)+
     annotate("text",as_datetime('2019-02-01'),y=28.9,label="MMM +1°C",fontface="italic",color="darkgrey",size=2)+
     annotate("segment",x=as_datetime("2017-08-14"),xend=as_datetime("2017-09-20"),y=20,yend=20,color="black")+
     annotate("segment",x=as_datetime("2017-10-04"),xend=as_datetime("2017-10-23"),y=20,yend=20,color="lightgray")+
     annotate("segment",x=as_datetime("2017-10-23"),xend=as_datetime("2017-11-13"),y=20,yend=20,color="black")+
     annotate("segment",x=as_datetime("2017-11-13"),xend=as_datetime("2017-12-20"),y=20,yend=20,color="lightgray")+
     annotate("segment",x=as_datetime("2017-12-20"),xend=as_datetime("2018-02-01"),y=20,yend=20,color="black")+
     annotate("segment",x=as_datetime("2018-02-01"),xend=as_datetime("2018-05-09"),y=20,yend=20,color="lightgray")+
     annotate("segment",x=as_datetime("2018-05-09"),xend=as_datetime("2018-07-12"),y=20,yend=20,color="black")+
     annotate("text",x=as_datetime("2017-12-30"),y=20.6,label="Sediment Deployments", size=2, fontface="italic",color="darkgray")+
     annotate("segment",x=as_datetime("2019-03-18"),xend=as_datetime("2019-03-29"),y=20,yend=20,color="darkgray")+
     annotate("text",x=as_datetime("2019-02-15"),y=20.6,label="Wave Deployment", size=2, fontface="italic",color="darkgray")
     
list<-sed_plotdata%>%group_by(site,block)%>%mutate(median=median(sed,na.rm=TRUE))%>%select(site,block,median)%>%distinct()%>%
     group_by(block)%>%arrange(block,desc(median))

sed_plotdata$site <- factor(sed_plotdata$site,levels = list$site)

plotdata<-envirodata%>%select(block,site,overall_min,overall_max,depth_m,total_DHW)%>%
     gather(cat,temp,-depth_m,-total_DHW,-block,-site)%>%
     mutate(dhwplot=case_when(temp<25~0,temp>25~total_DHW))
plotdata$site <- factor(plotdata$site,levels = list$site)

temp_plot<-ggplot()+
     geom_line(aes(site,temp,group=site,size=depth_m),color="grey",data=plotdata)+
     geom_point(aes(site,temp,fill=dhwplot,size=depth_m,),pch=21,data=plotdata%>%filter(temp>24))+
     geom_point(aes(site,temp,size=depth_m,),pch=21,data=plotdata%>%filter(temp<24),fill="grey")+
     theme_classic(base_size=8)+
     scale_fill_gradient(low="blue",high="red",breaks=c(0,1,2),limits=c(0,2.25))+
     theme(legend.position=c(0.15,0.97),
           legend.key.size=unit(0.2,"cm"),
           legend.spacing.y = unit(0.1, 'cm'),
           legend.spacing.x=unit(0.1,"cm"),
           axis.text.x=element_text(angle=90,vjust=0.5,hjust=1),
           axis.title.x=element_blank(),
           legend.key = element_blank(),
           legend.background = element_blank(),
           legend.box = "horizontal")+
     ylab("Temperature\nRange (hourly,°C)")+
     scale_size_continuous(range=c(1,5),breaks=c(0.5,1,3))+
     scale_y_continuous(limits=c(18,36),breaks=seq(18,36,3))+
     guides(fill = guide_colorbar(title = "DHW",
                                  direction="horizontal",
                                  label.position = "bottom",
                                  title.position = "top", 
                                  title.vjust = 0.5,
                                  label.vjust=0.5,
                                  barwidth = 3,
                                  barheight = 0.5),
            size=guide_legend(title="Depth (m)",
                              direction="horizontal",
                              label.position = "bottom",
                              title.position = "top", 
                              title.vjust = 0.5,
                              label.vjust=0.5,
                              nrow=1))

sed_plot<-ggplot(sed_plotdata)+geom_boxplot(aes(site,sed,fill=block),outlier.size = 0.5)+
     theme_classic(base_size=8)+
     ylab('Sedimentatio n\nRate (g/day)')+
     xlab("Site")+
     scale_fill_viridis_d(name="Block")+
     theme(legend.position="none",
           axis.text.x=element_text(angle=90,vjust=0.5,hjust=1))+
     annotate("text",label="7 deployments",x=2.5,y=2.3,fontface='italic',size=2,color="darkgrey")

wave_plot<-ggplot(waves_plotdata)+geom_boxplot(aes(block,rms,fill=block),width=1,outlier.size = 0.5)+
     scale_fill_viridis_d(name="Block")+
     theme_classic(base_size=8)+
     theme(legend.position='none',
           legend.key.size=unit(0.25,"cm"))+
     xlab("Block")+
     ylab("Mean Wave\nVelocity (cm/s)")

quartz(w=5.2,h=5.4)
plots<-cowplot::align_plots(timeseries,temp_plot,align="v",axis="l")
plot_grid(plot_grid(plots[[1]],pca,rel_widths=c(2,1),align="h",axis="tb",labels=c("A","B"),label_size=8),NULL,plots[[2]],NULL,sed_plot,wave_plot,ncol=1,align="h",axis="l",rel_heights=c(2,-0.3,1.3,-0.1,1,1),labels=c("","","C","","D","E"),label_size=8,label_y=c(0,0,1,1.1,1.1,1.1))

###################### IMPORT IBS DATA ############################################### #####
raw<-read_tsv("./data/coraldata/All_640_Samples_Sorted.IBS.ibsMat",col_names=FALSE)%>%select(-X641);rawdata<-raw[-c(637:638),-c(637:638)]
samples<-read_tsv("./data/coraldata/All_640_Sample_Names_Sorted.txt",col_names = FALSE)%>%filter(X1!='47B',X1!='479B')%>%mutate(sample=str_remove(X1,'B'))%>%select(-X1)
meta<-read_tsv("./data/coraldata/All_640_Samples_Site_Info_sortedBySample.txt")%>%clean_names()%>%mutate(sample=as.factor(sample))%>%filter(sample!='47B',sample!='479B')%>%mutate(sample=str_remove(sample,'B'))
depth<-read_tsv("./data/environmentaldata/depth.txt")%>%clean_names()
colnames(rawdata)<-as.vector(samples$sample)
###################### DETERMINE THRESHOLD FROM REPLICATES ########################### #####
replicates<-rawdata[1:69,1:69]
matrix<-as.matrix(replicates)
matrix[upper.tri(matrix)]<-NA;diag(matrix)=NA
replicate_distances<-as.data.frame(matrix)%>%gather(sample,dist)%>%drop_na()
replist<-as.data.frame(colnames(replicates))

#ggplot(replicate_distances)+geom_histogram(aes(dist)) #clear cut between replicates (<0.1) and cross-replicate comparisons (>0.1), choosing cuoff of 0.075 to eliminate janky replicate values
cut<-quantile((replicate_distances%>%filter(dist<0.075))$dist, c(.95));cut
plotdata<-replicate_distances%>%filter(dist<0.1)%>%mutate(group=case_when(dist>0.068~'Outlier',TRUE~'yes'))

dist<-ggplot(plotdata,aes(x=dist))+
     geom_histogram(aes(y = ..count..),bins=100)+
     geom_density(aes(y = ..density../5,fill=group),alpha=0.25)+
     scale_fill_manual(values=c("red","gray"))+
     theme_classic(base_size=8)+
     ylab('Count')+
     theme(legend.position="none",axis.title.y=element_blank())+
     #scale_y_reverse()+
     scale_x_continuous(limits=c(0.03,0.18),breaks=seq(0.02,0.18,0.02))+
     coord_flip()+
     annotate("text",x=.1,y=15,label="outliers",size=2,color="red",fontface="italic")+
     annotate("text",x=.07,y=30,label="threshold",size=2,color="gray",fontface="italic")+
     geom_vline(xintercept=cut,linetype="dashed",color="gray");dist
     
###################### CLONALITY ANALYSIS  ########################################### #####
matrix<-as.matrix(rawdata)
matrix[upper.tri(matrix)]<-NA;diag(matrix)=NA
cluster<-hclust(as.dist(matrix),method="complete")
plot(cluster)
clones<-rect.hclust(cluster,h=cut)
cloneslist<-do.call(rbind, lapply(seq_along(clones), function(i)
     {data.frame(CLUSTER=i, clones[[i]])
}))

summary_clonality<-cloneslist%>%rownames_to_column(var='sample')%>%select(-3)%>%left_join(.,meta,by="sample")%>%
     select(sample,CLUSTER,site)%>%
     filter(!str_detect(sample,"a|b"))%>%
     mutate(CLUSTER=as.factor(CLUSTER))%>%
     group_by(CLUSTER)%>%
     mutate(sites=n_distinct(site))
#write.table(summary_clonality,"/data/coraldata/summary_clonality.txt",sep="\t",quote=FALSE,row.names=FALSE)

summary<-cloneslist%>%rownames_to_column(var='sample')%>%select(-3)%>%clean_names()%>%inner_join(.,meta,by="sample")%>%
     filter(!str_detect(sample, "a|b"))%>%   #remove replicates
     group_by(site)%>%
     summarise(clusters = n_distinct(cluster),
               samples=n_distinct(sample))%>%
     rename(genotypes=clusters)%>%
     mutate(G_R=genotypes/samples)%>%
     separate(site,into=c("block","trash"),sep="_",remove=FALSE)

clustercounts<-summary_clonality%>%select(CLUSTER)%>%group_by(CLUSTER)%>%tally()
summary_clonality%>%filter(CLUSTER==352)

# enviro<-readRDS('./data/environmentaldata/environmental_summary')%>%drop_na()%>%semi_join(.,rawgenetic,by='sample')%>%arrange(sample)%>%select(-sed_sd,-sed_max,-rms_sd,-height_sd,-overall_mean,-overall_sd)%>%select(site,depth_m)
# depth_comp<-summary_clonality%>%group_by(CLUSTER)%>%add_tally()%>%ungroup()%>%left_join(.,enviro,by="site")%>%
#   mutate(group=case_when(n>1~"clonal",TRUE~"non"))%>%distinct()
# table(depth_comp$group)
# 
# depth_comp%>%group_by(group)%>%summarise(mean=mean(depth_m))
# wilcox.test(depth_m~group,data=depth_comp)



###################### CLONALITY FIGURE ############################################## #####
working<-cloneslist%>%rownames_to_column(var='sample')%>%#select(-3)%>%
     left_join(.,cloneslist%>%group_by(CLUSTER)%>%tally()%>%filter(n>1),by="CLUSTER")%>%
     #mutate(group=case_when(grepl("a|b|^347|^397|^90|^234|^596|^497|^146|^254|^146|6a|6b|476", sample)~ 'Biological Replicates',
     mutate(group=case_when(grepl("a|b|^347|^596|^497|^254|6a|6b|476", sample)~ 'Biological Replicates',
                            n>1~"Inferred Clones",
                            TRUE ~ "Sample"))%>%
     rename(label=sample)%>%#select(label,group)%>%
     mutate(group=case_when(label=='365'~'Sample',
                            TRUE~as.character(group)))

hcdata<-dendro_data(hclust(as.dist(matrix),method="complete"),type="rectangle")
hcdata$segments<-hcdata$segments%>%mutate(yend=case_when(yend==0~(y-0.01),TRUE ~ (yend)))
hcdata$segments<-left_join(hcdata$segments,hcdata$label,by="x")%>%select(-y.y)%>%rename(y=y.x)%>%left_join(.,working,by="label")

tree<-ggplot()+ 
     geom_hline(yintercept=cut,linetype="dashed",color="gray")+
     geom_segment(data=segment(hcdata), aes(x=x, y=y, xend=xend, yend=yend,color=group),size=0.5)+
     scale_color_discrete(limits=c('Biological Replicates','Inferred Clones'),name=element_blank())+
     theme_classic(base_size=8)+
     theme(axis.text.x=element_blank(),
           axis.ticks.x=element_blank(),
           axis.title.x=element_blank())+
     ylab("Genetic Distance (1-IBS)")+
     scale_y_continuous(limits=c(0.03,0.18),breaks=seq(0.02,0.18,0.02))+
     annotate("text",x=85,y=.042,label="BR1",size=2,fontface="italic")+
     annotate("text",x=130,y=0.04,label="BR2",size=2,fontface="italic")+
     annotate("text",x=190,y=.038,label="BR3",size=2,fontface="italic")+
     annotate("text",x=220,y=0.04,label="BR4",size=2,fontface="italic")+
     annotate("text",x=240,y=0.041,label="BR5",size=2,fontface="italic")+
     annotate("text",x=360,y=0.04,label="BR6",size=2,fontface="italic")+
     annotate("text",x=400,y=0.04,label="BR7",size=2,fontface="italic")+
     annotate("text",x=505,y=0.042,label="BR8",size=2,fontface="italic")+
     #annotate("text",x=535,y=0.037,label="BR9",size=2,fontface="italic")+
     annotate("text",x=570,y=0.042,label="BR9",size=2,fontface="italic")+
     annotate("text",x=605,y=0.039,label="BR10",size=2,fontface="italic")+
     annotate("text",x=35,y=0.031,label="BR = biological replicate set",size=2,fontface='italic',color='black')+
     annotate("text",x=420,y=0.031,label="Site 5_6",size=2,fontface="italic")+
     annotate("segment",x=415,xend=420,y=0.039,yend=0.032,color="black")+
     theme(legend.key.size=unit(0.3,"cm"),
           legend.position=c(0.2,0.86));tree

quartz(w=7.2,h=3)
plot_grid(dist,tree,align="h",axis="tb",rel_widths=c(1,6),nrow=1,labels=c("A","B"),label_size=8)
###################### WITHIN SITE IBD ############################################### #####
site_list<-meta%>%select(site)%>%distinct()
mantel_output<-data.frame(mantel_p=numeric(),mantel_obs=numeric())

for (i in site_list$site){
     temp<-meta%>%filter(!str_detect(sample, "a|b"))%>%select(site,sample,x,y)%>%mutate(sample=str_remove(sample,'B'))%>%filter(site==paste0(i))%>%arrange(as.numeric(sample))
     phys_dist<-as.dist(dist(as.matrix(temp%>%select(x,y)),method="euclidean"))
     gen_dist<-as.dist(bind_cols(samples,rawdata)%>%
          semi_join(.,temp,by='sample')%>%arrange(as.numeric(sample))%>%
          select(all_of(as.character(temp$sample))))
     test<-mantel.rtest(gen_dist, phys_dist, nrepet = 99)
     mantel_output[i,1]<-test$pvalue
     mantel_output[i,2]<-test$obs
}

mantel_output$mantel_p<-p.adjust(mantel_output$mantel_p,method="fdr")
###################### WITHIN SITE GENETIC DISTANCE ################################## #####
wilcox_output<-data.frame(wilcox_gendist_p=numeric())
wilcox_site_list<-summary%>%filter(G_R<1)%>%select(site)%>%distinct()

for (i in wilcox_site_list$site){
     temp<-meta%>%filter(!str_detect(sample, "a|b"))%>%select(site,sample,x,y)%>%mutate(sample=str_remove(sample,'B'))%>%filter(site==paste0(i))%>%arrange(as.numeric(sample))
     cluster_designation<-cloneslist%>%rownames_to_column(var='sample')%>%select(-3)%>%clean_names()
     gen_dist<-bind_cols(samples,rawdata)%>%rename(sample=1)%>%
                       semi_join(.,temp,by='sample')%>%arrange(as.numeric(sample))%>%
                       select(sample,all_of(as.character(temp$sample)))%>%
          gather(sample2,dist,-sample)%>%
          filter(sample!=sample2)%>%distinct(dist,.keep_all=TRUE)%>%
          left_join(cluster_designation,by="sample")%>%
          left_join(cluster_designation%>%rename(sample2=sample),by="sample2")%>%
          rename(cluster=cluster.x,cluster2=cluster.y)%>%
          mutate(clones=case_when(cluster==cluster2~'clonal',
                             cluster!=cluster2~'unique'))
          wilcox_output[i,1]<-glance(wilcox.test(dist~clones,data=gen_dist,alternative='less'))$p.value
}

wilcox_output$wilcox_gendist_p<-p.adjust(wilcox_output$wilcox_gendist_p,method="fdr")


###################### WITHIN SITE RELATEDNESS ####################################### #####
site_list<-meta%>%select(site)%>%distinct()
IBS_output<-data.frame(mean_rel=numeric(),median_rel=numeric(),mean_rel_noclones=numeric(),median_rel_noclones=numeric())

for (i in site_list$site){
     temp<-meta%>%filter(!str_detect(sample, "a|b"))%>%select(site,sample,x,y)%>%mutate(sample=str_remove(sample,'B'))%>%filter(site==paste0(i))%>%arrange(as.numeric(sample))%>%
          left_join(.,cloneslist%>%rownames_to_column(var="sample"),by="sample")%>%select(-6)
     gen_dist<-bind_cols(samples,rawdata)%>%
                            semi_join(.,temp,by='sample')%>%arrange(as.numeric(sample))%>%
                            select(all_of(as.character(temp$sample)))
     matrix<-as.matrix(gen_dist)
     matrix[upper.tri(matrix)]<-NA;diag(matrix)=NA
     IBS_output[i,1]<-mean(matrix,na.rm=TRUE)
     IBS_output[i,2]<-median(matrix,na.rm=TRUE)
     
     temp_noclones<-temp%>%distinct(CLUSTER,.keep_all=TRUE)
     gen_dist<-bind_cols(samples,rawdata)%>%
          semi_join(.,temp_noclones,by='sample')%>%arrange(as.numeric(sample))%>%
          select(all_of(as.character(temp_noclones$sample)))
     matrix<-as.matrix(gen_dist)
     matrix[upper.tri(matrix)]<-NA;diag(matrix)=NA
     IBS_output[i,3]<-mean(matrix,na.rm=TRUE)
     IBS_output[i,4]<-median(matrix,na.rm=TRUE)
}

table2<-left_join(summary,mantel_output%>%rownames_to_column(var="site"),by="site")%>%
     left_join(.,wilcox_output%>%rownames_to_column(var="site"),by="site")%>%
     left_join(.,IBS_output%>%rownames_to_column(var="site"),by="site")%>%
     select(block,site,samples,genotypes,G_R,mantel_p,wilcox_gendist_p,mean_rel,mean_rel_noclones)%>%
     clean_names(case='title')%>%
     rename('G:R'=5, 'Wilcox Distance' =7,"Mean Relatedness"=8,"Mean Relatedness w/o Clones"=9)%>%
     mutate(across(5:9, round, 3))

#saveRDS(table2,"./coraldata/genetic_summary")
#write.table(table2,"./manuscript/table2.txt",sep="\t",quote=FALSE,row.names = FALSE)

###################### BAYWIDE IBD ################################################### #####
coords<-read_tsv("./data/metadata/Site_depth_lat_long_numColonies.txt")%>%clean_names()%>%
     select(site,lat,lon)%>%arrange(site)
geographic_dist<-as.data.frame(geodist(coords%>%select(lat,lon)));colnames(geographic_dist)<-coords$site
ibd<-bind_cols(coords$site,geographic_dist)%>%rename(site=1)%>%gather(site2,dist,-site)

mantel_working<-bind_cols(samples,rawdata)%>%rename(sample=1)%>%gather(sample2,dist,-sample)%>%
     inner_join(.,meta%>%filter(!str_detect(sample, "a|b")),by='sample')%>%drop_na()%>%
     select(sample,sample2,dist,site)%>%
     inner_join(.,meta%>%filter(!str_detect(sample, "a|b"))%>%rename(sample2=sample),by='sample2')%>%drop_na()%>%
     select(sample,sample2,dist,site.x,site.y)%>%rename(site=site.x,site2=site.y)%>%
     inner_join(.,ibd,by=c("site",'site2'))%>%
     filter(dist.x>0.075)

gen_dist<-dist(mantel_working%>%select(sample,sample2,dist.x)%>%spread(sample2,dist.x)%>%select(-sample))
phys_dist<-dist(mantel_working%>%select(sample,sample2,dist.y)%>%spread(sample2,dist.y)%>%select(-sample))
test<-mantel.rtest(gen_dist, phys_dist, nrepet = 999);test

quartz()
ggplot(mantel_working)+geom_point(aes(dist.y,dist.x),alpha=0.01,size=0.5)+
     geom_smooth(aes(dist.y,dist.x),method="lm",size=1)+
     xlab("Physical Distance (m)")+ylab("Genetic Distance (1-IBS)")+
     theme_classic(base_size=8)+
     annotate("text",x=9000,y=0.075,label="Mantel p=0.71",size=2,fontface="italic")

###################### IMPORT GL DATA ################################################ #####
#this chunk does not run because files output.beagle and output.geno exceed github file size limits
#processing steps yield a file called formatted_GLs which can be read in as an RDS object in the next section

# samples<-read_tsv("./data/coraldata/sample_order.txt",col_names = FALSE)%>%rename(coral=1)%>%rownames_to_column(var="n")%>%
#      mutate(n=as.numeric(n)-1)%>%mutate(sample=paste0('Ind',n))
# depth<-read_tsv("./data/coraldata/output.counts")[,-581]%>%mutate(across(where(is.numeric), ~na_if(.,0)))%>%mutate(mean=rowMeans(.[,1:580],na.rm=TRUE),missing=rowSums(is.na(.)))
# loci<-read_tsv("./data/coraldata/output.beagle")%>%select(marker,allele1,allele2)%>%bind_cols(depth$mean,depth$missing)%>%rename(mean_depth=4,missing_count=5)
# keepers<-loci%>%filter(mean_depth>=10,missing_count<100)
# cols<-colnames(read_tsv("./data/coraldata/output.beagle",name_repair = "universal")%>%select(marker,contains('Ind'))%>%clean_names())
# rawdat<-read_tsv("./data/coraldata/output.geno",col_names=FALSE)%>%select(-X1743)%>%unite(marker,X1,X2,sep="_")%>%semi_join(.,keepers,by='marker')
# colnames(rawdat)<-cols
# 
# format<-rawdat%>%select(marker,contains('_'))%>%gather(sample,GL,-marker)%>%
#      separate(sample,into=c("sample","rep"))%>%arrange(marker)%>%
#   group_by(marker,sample)%>%
#   mutate(rep=row_number())%>%
#   filter(rep!=1)%>%mutate(GL=case_when(rep==3~GL*2,TRUE~as.numeric(GL)))%>%
#      group_by(marker,sample)%>%summarise(GL=sum(GL))%>%
#      mutate(sample=str_to_sentence(sample))%>%
#      left_join(.,samples,by="sample")%>%select(marker,coral,GL)%>%
#      spread(coral,GL)
# 
# working<-as.data.frame(t(format))%>%row_to_names(row_number=1)%>%rownames_to_column(var="sample")%>%
#      mutate_at("sample", str_replace, "B", "")%>%
#      filter(sample!='576b1')%>%
#      mutate(sample=as.numeric(sample))%>%
#      arrange(sample)
# 
# saveRDS(working,"./data/coraldata/formatted_GLs")
# 
# mean(keepers$mean_depth)     
# sd(keepers$mean_depth)
# min(keepers$mean_depth);max(keepers$mean_depth)
# #9955 loci, average depth (20+/-8 (range 10-63))
# 480/580
# 577/580

###################### dbRDA ######################################################### #####
clones<-read_tsv("./data/coraldata/summary_clonality.txt")%>%select(sample,CLUSTER)
set.seed(3839);rawgenetic<-readRDS('./data/coraldata/formatted_GLs')%>%arrange(sample)%>%left_join(.,clones,by="sample")%>%select(sample,CLUSTER,everything())%>%
     group_by(CLUSTER)%>%sample_n(1)%>%ungroup()%>%select(-CLUSTER)
enviro<-readRDS('./data/environmentaldata/environmental_summary')%>%drop_na()%>%semi_join(.,rawgenetic,by='sample')%>%arrange(sample)%>%select(-sed_sd,-sed_max,-rms_sd,-height_sd,-overall_mean,-overall_sd)
genetic<-rawgenetic%>%semi_join(.,enviro,by='sample')

format_enviro<-as.data.frame(scale(enviro%>%select(-block,-site,-sample)))
format_genetic<-genetic%>%select(-sample)%>%mutate_all(as.numeric)
#rankindex(format_enviro, format_genetic, indices = c("euc", "man", "gow","bra", "kul"), stepacross= FALSE, method = "pearson") 
dbRDA<-capscale(format_genetic ~ . ,format_enviro, dist="bray")

anova(dbRDA)
set.seed(3839);modelout<-anova(dbRDA, by="terms", permu=999);modelout
stat_summary<-as.data.frame(modelout)%>%rownames_to_column(var="factor")%>%rename(p_value=5)%>%mutate(p_adj=p.adjust(.$p_value,method="bonferroni"))%>%
     filter()%>%mutate(sig=case_when(p_adj<=0.05~"*"))%>%arrange(desc(SumOfSqs))%>%
  mutate(var_contrib=SumOfSqs/38.71068);stat_summary

write.table(stat_summary,"./manuscript/table3.txt",sep="\t",quote=FALSE,row.names = FALSE)

1-(stat_summary$SumOfSqs[1]/sum(stat_summary$SumOfSqs))

plotdata <- as.data.frame(scores(dbRDA, display = "sites"))%>%bind_cols(enviro)%>%left_join(.,left_join(clones,clones%>%select(CLUSTER)%>%group_by(CLUSTER)%>%tally(),by='CLUSTER')%>%select(sample,n)%>%arrange(desc(n)),by="sample")%>%
     mutate(clonal=case_when(n>1~"Clonal",TRUE~"Unique Genotype"))
d <- data.frame(factor = rownames(dbRDA$CCA$biplot), dbRDA$CCA$biplot)%>%left_join(.,stat_summary%>%select(factor,SumOfSqs,p_adj),by="factor")%>%filter(p_adj<0.05)%>%
     arrange(desc(SumOfSqs))%>%
     select(factor,CAP1,CAP2)%>%mutate(CAP1=CAP1*5,CAP2=CAP2*5)%>%
     mutate(factor=case_when(factor=='sed_mean'~"Sedimentation",
                             factor=='depth_m'~"Depth",
                             factor=='height_mean'~'Wave Height',
                             factor=='overall_residual'~'Temp Residual',
                             factor=='hrs_above_30'~'Hrs >30°C',
                             factor=='summer_mean_daily_range'~'Sumer Daily Range',
                             factor=="total_DHW"~"DHW",
                             factor=='overall_min'~'Min Temp'))%>%
     rownames_to_column(var="variable")

set.seed(3839);adonis2(format_genetic ~ enviro$block, data = enviro,nperm=100) #PERMANOVA 
set.seed(411)
p1<-ggplot(plotdata)+
     geom_point(aes(CAP1,CAP2),size=1,alpha=0.5,color="white")+
     theme_classic(base_size=8)+
     geom_segment(data = d, aes(x = 0, y = 0, xend = (CAP1),yend = (CAP2)), arrow = arrow(length = unit(0.2, "cm")),color = "darkblue")+
     geom_label_repel(aes(CAP1,CAP2,label=variable),data=d%>%filter(CAP1<0),size=2,hjust=0.5,segment.color="lightgray",min.segment.length = 0,)+
     geom_label_repel(aes(CAP1,CAP2,label=variable),data=d%>%filter(CAP1>0),size=2,hjust=0.5,segment.color="lightgray",min.segment.length = 0)+
     scale_color_viridis_d(name="Block")+
     theme(legend.position=c(0.1,0.2),legend.key.size=unit(0.2,"cm"))+
     annotate("text",label="Only Significant Factors Listed (Bonferroni p<0.05)",size=2,fontface="italic",color="darkgrey",x=-3.5,y=4,hjust=0)+
     #annotate("text",label='1: Depth\n2: Sedimentation\n3: Wave Height\n4: Minimum Temp\n5: Residual Temp\n6: Hrs >30°C\n7: DHW\n8: Summer Daily Range',x=-3.5,y=3.5,color="black",size=2,hjust=0,vjust=1)+
     annotate("text",label='1: Depth\n2: Hrs >30°C\n3: Total DHW\n4: Mean Sedimentation\n5: Summer Daily Range\n6: Minimum Temp\n7: Wave Height\n8: Overall Residual',x=-3.5,y=3.5,color="black",size=2,hjust=0,vjust=1)+
     scale_y_continuous(limits=c(-2,4),breaks=seq(-2,4,1))+scale_x_continuous(limits=c(-3.5,3),breaks=seq(-4,3,1));p1

p2<-ggplot(plotdata)+
     geom_segment(data = d, aes(x = 0, y = 0, xend = (CAP1),yend = (CAP2)), arrow = arrow(length = unit(0.2, "cm")),color = "darkgrey")+
     stat_ellipse(aes(CAP1,CAP2,group=block,color=block))+
     geom_point(aes(CAP1,CAP2,color=block),size=1)+
     theme_classic(base_size=8)+
     scale_color_viridis_d(name="Block")+
     theme(legend.position=c(0.1,0.75),legend.key.size=unit(0.2,"cm"))+
     annotate("text",label="Single Representative of Each Genotype",size=2,fontface="italic",color="darkgrey",x=-3.5,y=4,hjust=0)+
     annotate("text",label="PERMANOVA ~block p<0.001",size=2,fontface="italic",color="darkgrey",x=-3.5,y=-2,hjust=0)+
     scale_y_continuous(limits=c(-2,4),breaks=seq(-2,4,1))+scale_x_continuous(limits=c(-3.5,3),breaks=seq(-4,3,1))+
     xlab('CAP1')+ylab('CAP2')


p3<-ggplot(plotdata)+
     #stat_ellipse(aes(CAP1,CAP2,group=block,color=block))+
     geom_point(aes(CAP1,CAP2,color=clonal),size=1,data=plotdata)+
     geom_point(aes(CAP1,CAP2),size=1,alpha=1,data=plotdata%>%filter(clonal=="Clonal"),color="turquoise4")+
     theme_classic(base_size=8)+
     #geom_segment(data = d, aes(x = 0, y = 0, xend = (CAP1*5),yend = (CAP2*5)), arrow = arrow(length = unit(1/2, "picas")),color = "black")+
     #geom_label_repel(aes(CAP1*5,CAP2*6,label=factor),data=d,size=3)+
     theme(legend.position=c(0.25,0.85),legend.key.size=unit(0.2,"cm"),legend.title=element_blank())+
     scale_color_manual(values=c("turquoise4","lightgrey"))+
     annotate("text",label="Single Representative of Each Genotype",size=2,fontface="italic",color="darkgrey",x=-3.5,y=4,hjust=0)+
     scale_y_continuous(limits=c(-2,4),breaks=seq(-2,4,1))+scale_x_continuous(limits=c(-3.5,3),breaks=seq(-4,3,1));p3

quartz(w=7.2,h=2.3)
plot_grid(p1,p2,p3,nrow=1,labels='AUTO',label_size=8)

###################### WAVE ENERGY CAUSES FRAGMENTATION ############################## #####
enviro<-readRDS("./data/environmentaldata/environmental_summary")%>%select(-sample)%>%distinct()
genetic<-readRDS("./data/coraldata/genetic_summary")%>%clean_names()
out<-full_join(enviro,genetic,by="site")

library(plotrix)
fig<-out%>%select(block.x,g_r,height_mean)%>%rename(block=block.x)%>%group_by(block)%>%summarise(g_rmean=mean(g_r),g_rse=std.error(g_r),height=mean(height_mean))

ggplot(fig)+geom_point(aes(height,g_rmean))+geom_errorbar(aes(height,ymin=g_rmean-g_rse,ymax=g_rmean+g_rse),width=0)+
     theme_classic()+
     geom_smooth(aes(height,g_rmean),method="lm")

summary(lm(g_r~height_mean+depth_m,data=out))
summary(lm(g_r~rms_mean,data=out))


###################### DATA MAPS ##################################################### #####
library(sf);library(ggrepel);library(ggsn);library(cowplot);library(tidyverse);library(patchwork);library(jpeg);library(ggnewscale);library(spdep)

oahu<-st_transform(st_read("./data/oahu_map/coast_n83.shp"),crs=4326)
cropped<-st_crop(oahu,xmin=-158.8,xmax=-157.4,ymin=20.8,ymax=21.8)
fringe<-st_read("./data/oahu_map/Fringing Reef.shp")%>%select(id,geometry)%>%mutate(zone='Fringing Reef')
habitat<-st_read("./data/oahu_map/haw_benthic_habitat.shp")%>%clean_names()%>%filter(zone=='Backreef'|zone=="Reef Flat")%>%select(id,zone,geometry)
patch<-st_read("./data/oahu_map/Patches2.shp")%>%mutate(type="Reef")%>%clean_names()%>%select(id,zone,block,geometry)%>%mutate(zone='Patch Reef')
#points<-read_tsv("./metadata/Site_depth_lat_long_numColonies.txt")%>%clean_names()%>%select(site,block,lat,lon)%>%mutate(block=paste("Block",as.factor(block)))
enviro<-readRDS("./data/environmentaldata/environmental_summary")%>%select(-sample)%>%distinct()
stats<-read_tsv("./data/coraldata/genetic_stats.txt")%>%clean_names()
waveloggers<-read_tsv("./data/environmentaldata/wave_loggers.txt")%>%clean_names()%>%filter(site!='KB6')%>%separate(site,into=c('kb','block'),sep=2)%>%left_join(.,enviro%>%select(block,height_mean,rms_mean)%>%distinct())
points<-read_xlsx("./data/environmentaldata/kaneoheclopoints.xlsx")%>%clean_names()%>%rename(lat=3,lon=4)
     
meta_points<-left_join(left_join(points,stats,by='site'),enviro,by='site')%>%select(-block.x,-block.y)%>%select(site,block,lat,lon,everything())
out<-bind_rows(st_transform(fringe,crs=4326),st_transform(habitat,crs=4326))%>%
     mutate(zone=case_when(zone=="Reef Flat"~ "Fringing Reef",
                           zone=="Reef Crest"~"Fringing Reef",
                           zone=="Backreef"|zone=="Fringing Reef"~"Back/Fringing Reef",
                           TRUE~as.character(zone)))

rot = function(a) matrix(c(cos(a), sin(a), -sin(a), cos(a)), 2, 2)
tcrop<-st_geometry(st_transform(cropped%>%filter(COASTLIN_1==17),crs=32604))
tpoints<-st_geometry(st_transform(st_as_sf(meta_points, coords = c("lon", "lat"),crs=4326),crs=32604))
tout<-st_geometry(st_transform(out,crs=32604))
tpatch<-st_geometry(st_transform(patch,crs=32604))
cntrdc = st_centroid(tcrop)
twv<-st_geometry(st_transform(st_as_sf(waveloggers, coords = c("lon", "lat"),crs=4326),crs=32604))

d2 = st_as_sf(((tcrop - cntrdc) *rot(0.75) *2 + cntrdc))
p2 = st_as_sf(((tpoints - cntrdc) *rot(0.75) *2 + cntrdc))%>%bind_cols(.,meta_points)
t2 = st_as_sf(((tout - cntrdc) *rot(0.75) *2 + cntrdc))
pat2= st_as_sf(((tpatch - cntrdc) *rot(0.75) *2 + cntrdc))%>%bind_cols(.,patch)
w2 = st_as_sf(((twv - cntrdc) *rot(0.75) *2 + cntrdc))%>%bind_cols(.,waveloggers)

ex<-as.data.frame((st_geometry(p2,x)))%>%separate(geometry,into=c('x','y'),sep=",")%>%
     mutate(x=str_replace(x,"c\\(",""),y=str_replace(y,"\\)",""))%>%bind_cols(meta_points)

a<-ggplot()+
     geom_sf(data=t2,fill="lightgray",color="lightgray")+
     geom_sf(data=d2,fill="darkgray")+
     geom_sf(aes(fill=block),dat=pat2,color=NA)+
     theme_classic(base_size=8)+
     coord_sf(xlim=c(627000,636000),ylim=c(2338000,2365000))+
     theme(legend.position=c(0.2,0.9),
           legend.key.size=unit(0.2,"cm"),
           axis.ticks=element_blank(),
           axis.text=element_blank(),
           axis.title=element_blank())+
     scale_fill_viridis_d(name="Block")+
     annotate("text",x=634700,y=2364000,label="N",size=3)+
     geom_segment(aes(x=635000,xend=636200,y=2364500,yend=2366000),arrow = arrow(length = unit(0.2,"cm")))

alt<-ggplot()+
     geom_sf(data=t2,fill="lightgray",color="lightgray")+
     geom_sf(data=d2,fill="darkgray")+
     geom_sf(fill="lightgray",dat=pat2,color=NA)+
     geom_sf(aes(fill=block,size=depth_m),data=p2,pch=21,color="black")+
     theme_classic(base_size=8)+
     coord_sf(xlim=c(627000,636000),ylim=c(2338000,2365000))+
     theme(legend.position=c(0.25,0.83),
           legend.key.size=unit(0.2,"cm"),
           axis.ticks=element_blank(),
           axis.text=element_blank(),
           axis.title=element_blank(),
           legend.spacing=unit(0,"cm"),legend.box = "vertical")+
     scale_fill_viridis_d(name="Block")+
     scale_size_continuous(range=c(1,5),breaks=c(0.5,1,3))+
     #annotate("text",x=634700,y=2364000,label="N",size=3)+
     #geom_segment(aes(x=635000,xend=636200,y=2364500,yend=2366000),arrow = arrow(length = unit(0.2,"cm")))+
     guides(size=guide_legend(title="Depth (m)",
                              direction="horizontal",
                              label.position = "bottom",
                              title.position = "top", 
                              title.vjust = 0,
                              nrow=1,order=1),
            fill = guide_legend(title = "Block",
                                direction="Horizontal",
                                label.position = "right",
                                title.position = "top", 
                                title.vjust = 0.5,
                                label.vjust=0.5,
                                barwidth = 3,
                                barheight = 0.5,order=1));alt

b<-ggplot()+
     geom_sf(data=t2,fill="lightgray",color="lightgray")+
     geom_sf(data=d2,fill="darkgray")+
     geom_sf(fill="lightgray",dat=pat2,color=NA)+
     geom_sf(aes(fill=g_r),data=p2,size=2,pch=21,color="black")+
     theme_classic(base_size=8)+
     coord_sf(xlim=c(627000,636000),ylim=c(2338000,2365000))+
     theme(legend.position=c(0.16,0.9),
           legend.key.size=unit(0.2,"cm"),
           axis.title=element_blank(),
           axis.ticks=element_blank(),
           axis.text=element_blank())+
     scale_fill_gradient(low="white",high="blue",name="G:R")

c<-ggplot()+
     geom_sf(data=t2,fill="lightgray",color="lightgray")+
     geom_sf(data=d2,fill="darkgray")+
     geom_sf(fill="lightgray",dat=pat2,color=NA)+
     geom_sf(aes(fill=mean_relatedness),data=p2,size=2,pch=21,color="black")+
     theme_classic(base_size=8)+
     coord_sf(xlim=c(627000,636000),ylim=c(2338000,2365000))+
     theme(legend.position=c(0.2,0.9),
           legend.key.size=unit(0.2,"cm"),
           axis.title=element_blank(),
           axis.ticks=element_blank(),
           axis.text=element_blank())+
     scale_fill_gradient(low="white",high="orange",name="Relat")

d<-ggplot()+
     geom_sf(data=t2,fill="lightgray",color="lightgray")+
     geom_sf(data=d2,fill="darkgray")+
     geom_sf(fill="lightgray",dat=pat2,color=NA)+
     geom_sf(aes(fill=height_mean),data=w2,size=5,pch=21,color="black")+
     theme_classic(base_size=8)+
     coord_sf(xlim=c(627000,636000),ylim=c(2338000,2365000))+
     theme(legend.position=c(0.18,0.9),
           legend.key.size=unit(0.2,"cm"),
           axis.title=element_blank(),
           axis.ticks=element_blank(),
           axis.text=element_blank())+
     scale_fill_gradient(low="white",high="darkgreen",name="Wave\nHeight (m)",breaks=seq(0,0.3,0.1),limits=c(0,0.35))

e<-ggplot()+
     geom_sf(data=t2,fill="lightgray",color="lightgray")+
     geom_sf(data=d2,fill="darkgray")+
     geom_sf(fill="lightgray",dat=pat2,color=NA)+
     geom_sf(aes(fill=hrs_above_30),data=p2,size=2,pch=21,color="black")+
     theme_classic(base_size=8)+
     coord_sf(xlim=c(627000,636000),ylim=c(2338000,2365000))+
     theme(legend.position=c(0.16,0.9),
           legend.key.size=unit(0.2,"cm"),
           axis.title=element_blank(),
           axis.ticks=element_blank(),
           axis.text=element_blank())+
     scale_fill_gradient(low="blue",high="red",name="Hrs Over\n30°C",breaks=seq(0,150,50),limits=c(0,150))

f<-ggplot()+
     geom_sf(data=t2,fill="lightgray",color="lightgray")+
     geom_sf(data=d2,fill="darkgray")+
     geom_sf(fill="lightgray",dat=pat2,color=NA)+
     geom_sf(aes(fill=overall_residual),data=p2,size=2,pch=21,color="black")+
     theme_classic(base_size=8)+
     coord_sf(xlim=c(627000,636000),ylim=c(2338000,2365000))+
     theme(legend.position=c(0.16,0.9),
           legend.key.size=unit(0.2,"cm"),
           axis.title=element_blank(),
           axis.ticks=element_blank(),
           axis.text=element_blank())+
     scale_fill_gradient2(low="blue",mid="white",high="red",name="Residual",midpoint=0)

quartz(w=7.2,h=3.5)
plot_grid(alt,b,c,d,e,f,nrow=1,labels='AUTO',label_size=8,label_y=c(0.09),label_x=c(0.06))




