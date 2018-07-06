#批量读入数据1------------------------------------------------------------------
path <- "C:/Users/huangsong.kh.AP/Desktop/4.12/pictures" ##文件目录
fileNames <- dir(path)  ##获取该路径下的文件名
filePath <- sapply(fileNames, function(x){ 
  paste(path,x,sep='/')})   ##生成读取文件路径
data <- lapply(filePath, function(x){
  read.table(x, header=F)})  ##读取数据，结果为list
l = length(data)
PictureName = unlist(strsplit(fileNames,'.t'))[seq(1,2*l,2)]

data1 = NULL
for(i in 1:l)
{
  pic_ind = data[[i]]
  len = dim(pic_ind)[1]
  pname = rep(PictureName[i],len)
  pic_ind = cbind(pname,pic_ind)
  data1 = rbind(data1,pic_ind)
}
names(data1) = c("pname","X","Y")

#批量读入数据2-----------------------------------------------------------------------
path <- "C:/Users/huangsong.kh.AP/Desktop/4.12/coord" ##文件目录
fileNames <- dir(path)  ##获取该路径下的文件名
filePath <- sapply(fileNames, function(x){ 
  paste(path,x,sep='/')})   ##生成读取文件路径
data <- lapply(filePath, function(x){
  read.table(x, header=F)})  ##读取数据，结果为list
l2 = length(data)
PictureName = unlist(strsplit(fileNames,'.t'))[seq(1,2*l2,2)]

data2 = NULL
for(i in 1:l2)
{
  pic_ind = data[[i]]
  len = dim(pic_ind)[1]
  pname = rep(PictureName[i],len)
  pic_ind = cbind(pname,pic_ind)
  data2 = rbind(data2,pic_ind)
}

#数据集合并----------------------------------------------------------------
data3 = merge(data1,data2,by = "pname",all.y = T)
picture = unique(data3$pname)
l2 = length(unique(picture))

data4 = list(NULL)
for(i in 1:l2)
{
  pic_ind =  data3[data3$pname==picture[i],]
  pic_ind$X = (pic_ind$X-pic_ind$V1)/pic_ind$V5
  pic_ind$Y = (pic_ind$Y-pic_ind$V2)/pic_ind$V6
  pic_ind = list(pic_ind[,c(2,3)])
  names(pic_ind) = picture[i]
  data4 = cbind(data4,pic_ind)
  if(i == l2) data4 = data4[-1]
}

#批量保存txt--------------------------------------------------------------------
outPath <- "C:/Users/huangsong.kh.AP/Desktop/4.12/new_pictures" ##输出路径
out_fileName <- sapply(picture,function(x){paste(x, ".txt", sep='')}) 
##txt格式
out_filePath  <- sapply(out_fileName, function(x){paste(outPath ,x,sep='/')}) ##输出路径名
##输出文件
for(i in 1:l2){
  write.table(data4[[i]], file=out_filePath[i], row.name=F,col.names = F) 
}

#-----------------------------------------------------------------