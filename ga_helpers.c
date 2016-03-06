int GAFlipCoin(float p)
{
	float c;
	c = ((float)rand())/RAND_MAX; 
	if(c < p)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}
