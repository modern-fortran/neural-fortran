program example_montesinos_uni
 use mod_kinds,only:ik,rk
 use mod_network,only:network_type
 implicit none
 integer(ik)::ny1_tr,ny2_tr,nx1_tr,nx2_tr
 integer(ik)::ny1_ts,ny2_ts,nx1_ts,nx2_ts

 integer(ik)::batch_size,num_epochs

 real(rk),allocatable::y_tr(:,:),x_tr(:,:)
 real(rk),allocatable::y_ts(:,:),x_ts(:,:)

 type(network_type)::net

 call readfile('../data/montesinos_uni/y_tr.dat',ny1_tr,ny2_tr,y_tr)
 call readfile('../data/montesinos_uni/x_tr.dat',nx1_tr,nx2_tr,x_tr)
 
 !net=network_type([nx1_tr,50,50,ny1_tr],'relu')
 net=network_type([nx1_tr,50,50,ny1_tr])

 batch_size=30
 num_epochs=20

 !training
 call net%fit(x_tr,y_tr,3._rk,epochs=num_epochs,batch_size=batch_size)

 call net%sync(1)

 !validation
 call readfile('../data/montesinos_uni/y_ts.dat',ny1_ts,ny2_ts,y_ts)
 call readfile('../data/montesinos_uni/x_ts.dat',nx1_ts,nx2_ts,x_ts)

 if(this_image().eq.1)then
  write(*,*)'Correlation(s): ',corr_array(net%predict(x_ts),y_ts)
 endif

contains

subroutine readfile(filename,n,m,array)
 character(len=*),intent(in)::filename
 integer(ik),intent(out)::n,m
 real(rk),allocatable,intent(out)::array(:,:)

 integer(ik)::un,i,io

 open(newunit=un,file=filename,status='old',action='read')
 call numlines(un,m)
 call numcol(un,n) 

 allocate(array(n,m))
 rewind(un)
 do i=1,m
  read(un,*,iostat=io)array(:,i)
  if(io.ne.0)exit
 enddo
 close(un)
 
end subroutine

pure function corr_array(array1,array2) result(a)
 real(rk),intent(in)::array1(:,:),array2(:,:)
 real(rk),allocatable::a(:)
 
 integer(ik)::i,n
 
 n=size(array1,dim=1)
 
 allocate(a(n))
 a=0.0_rk
 do i=1,n
  a(i)=corr(array1(i,:),array2(i,:))
 enddo

end function

pure real(rk) function corr(array1,array2)
 real(rk),intent(in)::array1(:),array2(:)
 
 real(rk)::mean1,mean2

 !brute force
 
 mean1=sum(array1)/size(array1)
 mean2=sum(array2)/size(array2)
 corr=dot_product(array1-mean1,array2-mean2)/sqrt(sum((array1-mean1)**2)*sum((array2-mean2)**2))

end function

subroutine numlines(unfile,n)
 implicit none
 integer::io
 integer,intent(in)::unfile
 integer,intent(out)::n
 rewind(unfile)
 n=0
 do
  read(unfile,*,iostat=io)
  if (io.ne.0) exit
  n=n+1
 enddo
 rewind(unfile)
end subroutine

subroutine numcol(unfile,n)
 implicit none
 integer,intent(in)::unfile
 character(len=1000000)::a
 integer,intent(out)::n
 integer::curr,first,last,lena,stat,i
 rewind(unfile)
 read(unfile,"(a)")a
 curr=1;lena=len(a);n=0
 do
  first=0
  do i=curr,lena
   if (a(i:i) /= " ") then
    first=i
    exit
   endif
  enddo
  if (first == 0) exit
  curr=first+1
  last=0
  do i=curr,lena
   if (a(i:i) == " ") then
    last=i
    exit
   endif
  enddo
  if (last == 0) last=lena
  n=n+1
  curr=last+1
 enddo
 rewind(unfile)
end subroutine

end program
