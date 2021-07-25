    public void bad() throws Throwable
    {
        int [] dataCopy;
        {
            int [] data;

            /* POTENTIAL FLAW: data is null */
            data = null;

            dataCopy = data;
        }
        {
            int [] data = dataCopy;

            /* POTENTIAL FLAW: null dereference will occur if data is null */
            IO.writeLine("" + data.length);

        }
    }
