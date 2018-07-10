library(ggplot2)
library(grid)


setwd("E:/graduate/class/Statistical Case Studies/Coauthor Networkcsv/杂志词频/")
files_name = dir(getwd())[c(1,4,6,9)]


theme_opts<-list(theme(axis.text.x=element_text(),
                       plot.title = element_text(hjust = 0.5,size=10,color="blue"),
                       panel.background=element_rect(fill='aliceblue',color='black'),
                       panel.grid.minor = element_blank(),
                       panel.grid.major =element_blank(),
                       plot.background = element_rect(fill="ivory1")))




vp <- function(x, y) {
  viewport(layout.pos.row = x, layout.pos.col = y)
}
grid.newpage()
num_row = 2 #调几行几列
num_col = 2
pushViewport(viewport(layout = grid.layout(num_row, num_col)))

#频数
for(i in 1:4)
{
  word_count = read.csv(files_name[i],header  =F,nrows = 20,skipNul = T)[seq(1,19,2),]
  maga = substr(files_name[i],12,(nchar(files_name[i])-4))
  p = ggplot(word_count, aes(x = reorder(V1,-V2), y = V2,fill='lightpink1')) +
    geom_bar(stat = "identity",width = 0.7,alpha=0.6) +
    scale_x_discrete("单词") +
    scale_y_continuous("词频") +
    ggtitle(paste(maga,"top10高频词",sep = " "))+
    guides(fill=FALSE)+
    annotate(geom="text",x = word_count$V1,y=word_count$V2,label=as.character(round(word_count$V2,2)),size=3,vjust = -0.5)+
    theme_opts
  if(i%%num_col==0){r = i/num_col;l = num_col}
  else{r = ceiling(i/num_col);l = i%%num_col}
  print(c(r,l))

  print(p,vp = vp(r, l))
  
}

#频率
num_paper = c(1439,522,709,664)
for(i in 1:4)
{
  word_count = read.csv(files_name[i],header  =F,nrows = 20,skipNul = T)[seq(1,19,2),]
  word_count$V2 = word_count$V2/num_paper[i]
  maga = substr(files_name[i],12,(nchar(files_name[i])-4))
  p = ggplot(word_count, aes(x = reorder(V1,-V2), y = V2,fill='lightpink1')) +
    geom_bar(stat = "identity",width = 0.7,alpha=0.6) +
    scale_x_discrete("单词") +
    scale_y_continuous("词频") +
    ggtitle(paste(maga,"top10高频词",sep = " "))+
    guides(fill=FALSE)+
    annotate(geom="text",x = word_count$V1,y=word_count$V2,label=as.character(round(word_count$V2,2)),size=3,vjust = -0.5)+
    theme_opts
  if(i%%num_col==0){r = i/num_col;l = num_col}
  else{r = ceiling(i/num_col);l = i%%num_col}
  print(c(r,l))
  
  print(p,vp = vp(r, l))
  
}